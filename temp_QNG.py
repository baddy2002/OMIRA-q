import torch
import torch.nn as nn
from torch.func import jacrev, vmap

# =====================================================================
# 1. DEFINIZIONE DELL'EVOLUZIONE DI RYDBERG DIFFERENZIABILE (1 GRAFO)
# =====================================================================
class RydbergHamiltonianSimulator:
    def __init__(self, n_atoms: int, n_steps: int, dt: float, C6: float, device="cuda"):
        self.n_atoms = n_atoms
        self.n_steps = n_steps
        self.dt = dt
        self.C6 = C6
        self.dim = 2 ** n_atoms
        self.device = device

        # Pre-calcolo degli operatori di Pauli e numero n_i nella base computazionale
        # n_i: matrice diagonale (0 o 1 per ogni stato di base) -> shape [n_atoms, dim]
        basis = torch.arange(self.dim, device=device).unsqueeze(1)
        bits = (basis >> torch.arange(n_atoms - 1, -1, -1, device=device)) & 1
        self.n_ops = bits.float().T  # [n_atoms, dim]

        # Somma globale sigma_x e sigma_y per l'impulso laser globale
        self.H_x, self.H_y = self._build_global_drive_ops()

        # Stato iniziale |00...0>
        self.psi0 = torch.zeros(self.dim, dtype=torch.complex64, device=device)
        self.psi0[0] = 1.0 + 0.0j

    def _build_global_drive_ops(self):
        sx = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64, device=self.device)
        sy = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex64, device=self.device)
        eye = torch.eye(2, dtype=torch.complex64, device=self.device)

        Hx = torch.zeros((self.dim, self.dim), dtype=torch.complex64, device=self.device)
        Hy = torch.zeros((self.dim, self.dim), dtype=torch.complex64, device=self.device)

        for a in range(self.n_atoms):
            ops_x = [sx if i == a else eye for i in range(self.n_atoms)]
            ops_y = [sy if i == a else eye for i in range(self.n_atoms)]
            kx, ky = ops_x[0], ops_y[0]
            for i in range(1, self.n_atoms):
                kx = torch.kron(kx, ops_x[i])
                ky = torch.kron(ky, ops_y[i])
            Hx += kx
            Hy += ky
        return Hx, Hy

    def evolve_state(self, z: torch.Tensor):
        """
        Evoluzione di Schrödinger per un singolo grafo.
        z: vettore 1D di parametri fisici [x_1..x_N, y_1..y_N, omegas(K), detunings(K), phase]
        Restituisce (psi.real, psi.imag) separati per compatibilità piena con jacrev.
        """
        N = self.n_atoms
        coords = z[: 2 * N].view(N, 2)  # [N, 2]
        pulse_params = z[2 * N :]
        n_segments = (pulse_params.shape[0] - 1) // 2
        omegas = pulse_params[:n_segments]
        detunings = pulse_params[n_segments : 2 * n_segments]
        phase = pulse_params[-1]

        # 1. Termine di interazione Van der Waals (diagonale nello spazio di Hilbert)
        # V_ab = C6 / r_ab^6
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 2]
        dist_sq = (diff ** 2).sum(dim=-1) + torch.eye(N, device=z.device) * 1e12
        V_matrix = self.C6 / (dist_sq ** 3)  # [N, N]
        V_matrix = torch.triu(V_matrix, diagonal=1)

        # Energia diagonale VdW per ognuno dei 2^N stati di base
        # sum_{a<b} V_ab * n_a * n_b
        H_vdw_diag = torch.einsum("ab,ad,bd->d", V_matrix, self.n_ops, self.n_ops)  # [dim]

        # 2. Integrazione temporale (metodo del punto medio / Runge-Kutta 2° ordine)
        psi = self.psi0
        steps_per_seg = self.n_steps // n_segments

        for step in range(self.n_steps):
            seg_idx = min(step // steps_per_seg, n_segments - 1)
            omega_t = omegas[seg_idx]
            delta_t = detunings[seg_idx]

            # Parte diagonale totale: VdW - Delta(t) * sum_a n_a
            H_diag = H_vdw_diag - delta_t * self.n_ops.sum(dim=0)

            # Parte off-diagonale: (Omega(t)/2) * (cos(phi) Hx - sin(phi) Hy)
            H_off = 0.5 * omega_t * (torch.cos(phase) * self.H_x - torch.sin(phase) * self.H_y)

            # Passo RK2 (molto più accurato di Eulero per evoluzioni unitarie)
            k1 = -1.0j * (H_diag * psi + H_off @ psi)
            psi_mid = psi + 0.5 * self.dt * k1
            k2 = -1.0j * (H_diag * psi_mid + H_off @ psi_mid)
            psi = psi + self.dt * k2

        # Normalizzazione per stabilità numerica su step finiti
        psi = psi / torch.linalg.norm(psi)
        return psi.real, psi.imag


# =====================================================================
# 2. LAYER CUSTOM CON FUBINI-STUDY ESATTA + REGOLARIZZAZIONE TIKHONOV
# =====================================================================
class QuantumNaturalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_batch: torch.Tensor, simulator: RydbergHamiltonianSimulator, tikhonov_lambda: float):
        """
        z_batch: [Batch, P] (parametri fisici snappati con STE)
        Output: probs [Batch, 2^N]
        """
        ctx.simulator = simulator
        ctx.tikhonov_lambda = tikhonov_lambda

        # Esecuzione vettorizzata su tutto il batch senza costruire il grafo di autodiff
        with torch.no_grad():
            psi_real, psi_imag = vmap(simulator.evolve_state)(z_batch)
            probs = psi_real ** 2 + psi_imag ** 2

        ctx.save_for_backward(z_batch)
        return probs

    @staticmethod
    def backward(ctx, grad_probs: torch.Tensor):
        """
        grad_probs: dL / d(probs) di shape [Batch, 2^N]
        Calcola:
          1) dL / dz (gradiente euclideo standard)
          2) g_ij (tensore metrico di Fubini-Study esatto via jacrev)
          3) Pre-condizionamento: (g + lambda * I)^(-1) @ (dL / dz)
        """
        (z_batch,) = ctx.saved_tensors
        simulator = ctx.simulator
        lam = ctx.tikhonov_lambda
        B, P = z_batch.shape

        # Funzione ausiliaria per un singolo elemento del batch:
        # restituisce (probs, psi_real, psi_imag)
        def single_sample_fn(z_single):
            pr, pi = simulator.evolve_state(z_single)
            probs = pr ** 2 + pi ** 2
            return probs, (pr, pi)

        # 1. Calcolo del Jacobiano dello stato d|psi>/dz via jacrev + vmap
        # has_aux=True permette di ottenere anche probs con un solo passaggio!
        # J_real, J_imag hanno shape: [Batch, 2^N, P]
        (J_real, J_imag), probs = vmap(
            jacrev(single_sample_fn, has_aux=True)
        )(z_batch)

        # Ricostruiamo psi e J in forma complessa
        with torch.no_grad():
            psi_real, psi_imag = vmap(simulator.evolve_state)(z_batch)
            psi = torch.complex(psi_real, psi_imag)          # [Batch, 2^N]
            J = torch.complex(J_real, J_imag)                # [Batch, 2^N, P]

            # 2. Calcolo del gradiente euclideo dL / dz = sum_k (dL/dp_k) * (dp_k/dz)
            # Poiché p_k = pr_k^2 + pi_k^2, dp_k/dz = 2 * (pr_k * d(pr_k)/dz + pi_k * d(pi_k)/dz)
            dp_dz = 2.0 * (
                psi_real.unsqueeze(-1) * J_real + psi_imag.unsqueeze(-1) * J_imag
            )  # [Batch, 2^N, P]
            grad_z = torch.einsum("bk,bkp->bp", grad_probs, dp_dz)  # [Batch, P]

            # 3. Calcolo esatto della metrica di Fubini-Study g_ij:
            # g_ij = Re[ <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> ]
            J_dag_J = torch.matmul(J.conj().transpose(1, 2), J)  # [Batch, P, P]
            J_dag_psi = torch.matmul(J.conj().transpose(1, 2), psi.unsqueeze(-1)).squeeze(-1)  # [Batch, P]
            overlap_term = torch.einsum("bi,bj->bij", J_dag_psi, J_dag_psi.conj())  # [Batch, P, P]

            Q = J_dag_J - overlap_term          # [Batch, P, P] Tensore Geometrico Quantistico complesso 
            g = Q.real                          # [Batch, P, P] Metrica di Fubini-Study (simmetrica)
            berry_F = -2.0 * Q.imag             # [Batch, P, P] Curvatura di Berry (antisimmetrica) 
            # 4. Regolarizzazione di Tikhonov con smorzamento levenberg-marquardt: (g + lambda * I)
            eye = torch.eye(P, device=z_batch.device, dtype=g.dtype).unsqueeze(0)
            # g_reg = g + lam * eye  # [Batch, P, P]
            # Invece di: g_reg = g + lam * eye
            # Prova a scalare lam sulla media degli autovalori (traccia / P) per adattarti al regime di blockade:
            trace_mean = torch.diagonal(g, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
            adaptive_lam = torch.clamp(0.05 * trace_mean, min=1e-5, max=1e-1)
            g_reg = g + adaptive_lam * eye

            gamma = 0.1  # Peso della curvatura di Berry per stabilizzare il pre-condizionamento
            eff_metric = g_reg + gamma * berry_F
            # 5. Risoluzione sistema lineare (g + lambda * I) @ nat_grad_z = grad_z
            nat_grad_z = torch.linalg.solve(eff_metric, grad_z.unsqueeze(-1)).squeeze(-1)  # [Batch, P]

        # Restituisce il gradiente pre-condizionato rispetto a z_batch (None per simulator e lambda)
        return nat_grad_z, None, None


# =====================================================================
# 3. ESEMPIO DI INTEGRAZIONE CON LA GNN E STE SULLA GRIGLIA
# =====================================================================
class HybridGNNQuantumClassifier(nn.Module):
    def __init__(self, node_feat_dim: int, n_atoms: int, n_segments: int, n_classes: int, grid_coords: torch.Tensor):
        super().__init__()
        self.n_atoms = n_atoms
        self.grid_coords = grid_coords  # [M_celle, 2] coordinate valide del RectangularLayout

        # GNN semplificata + Proiettore verso i P parametri fisici
        P = 2 * n_atoms + 2 * n_segments + 1
        self.gnn_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # Pool (Max + Avg) -> 128 feature -> P parametri fisici
        self.projector = nn.Linear(128, P)

        # Classificatore finale sulle probabilità dei 2^N bitstring
        self.readout = nn.Linear(2 ** n_atoms, n_classes)

        self.simulator = RydbergHamiltonianSimulator(
            n_atoms=n_atoms, n_steps=60, dt=0.05, C6=5420158.53, device=grid_coords.device
        )

    def snap_to_grid_ste(self, coords_raw: torch.Tensor) -> torch.Tensor:
        """
        Associa ciascun atomo alla cella libera più vicina nella griglia rettangolare
        usando Straight-Through Estimator (STE).
        coords_raw: [Batch, N, 2]
        """
        B, N, _ = coords_raw.shape
        coords_snapped = torch.zeros_like(coords_raw)

        for b in range(B):
            used_mask = torch.zeros(self.grid_coords.shape[0], dtype=torch.bool, device=coords_raw.device)
            for a in range(N):
                dists = torch.norm(self.grid_coords - coords_raw[b, a], dim=-1)
                dists = dists.masked_fill(used_mask, float("inf"))
                best_idx = torch.argmin(dists)
                used_mask[best_idx] = True
                coords_snapped[b, a] = self.grid_coords[best_idx]

        # STE: forward usa coords_snapped, backward passa il gradiente a coords_raw
        return coords_snapped.detach() + (coords_raw - coords_raw.detach())

    def forward(self, x_nodes: torch.Tensor, tikhonov_lambda: float = 1e-3):
        """
        x_nodes: [Batch, N_nodes_molecola, node_feat_dim]
        """
        h = self.gnn_encoder(x_nodes)
        h_pool = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)  # [Batch, 128]
        z_raw = self.projector(h_pool)  # [Batch, P]

        # Applichiamo vincoli fisici (es. STE sulle coordinate, softplus su ampiezze Omega >= 0)
        coords_raw = z_raw[:, : 2 * self.n_atoms].view(-1, self.n_atoms, 2)
        coords_valid = self.snap_to_grid_ste(coords_raw).view(-1, 2 * self.n_atoms)

        # Ricomponiamo z_valid mantenendo il grafo verso self.projector e self.gnn_encoder
        z_valid = torch.cat([coords_valid, z_raw[:, 2 * self.n_atoms :]], dim=-1)

        # Forward + Backward pre-condizionato con Fubini-Study esatta
        probs = QuantumNaturalLayer.apply(z_valid, self.simulator, tikhonov_lambda)
        logits = self.readout(probs)
        return logits


# =====================================================================
# 4. TEST RAPIDO DI ESECUZIONE SU GPU
# =====================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Griglia rettangolare 4x4 con passo 5.0 um
    gx, gy = torch.meshgrid(torch.arange(4, device=device) * 5.0, torch.arange(4, device=device) * 5.0, indexing="ij")
    grid_coords = torch.stack([gx.flatten(), gy.flatten()], dim=-1)

    model = HybridGNNQuantumClassifier(
        node_feat_dim=16, n_atoms=6, n_segments=3, n_classes=2, grid_coords=grid_coords
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Batch fittizio di 8 molecole
    x_dummy = torch.randn(8, 12, 16, device=device)
    y_dummy = torch.randint(0, 2, (8,), device=device)

    optimizer.zero_grad()
    logits = model(x_dummy, tikhonov_lambda=1e-3)
    loss = criterion(logits, y_dummy)
    loss.backward()  # Qui entra in azione QuantumNaturalLayer.backward con jacrev + Tikhonov!
    optimizer.step()

    print(f"Step completato su {device} | Loss: {loss.item():.4f}")