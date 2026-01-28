from .topology_to_instance import build_instance_from_topology
from .solver_gurobi import solve_gurobi
from .solver_gurobi_multiobjectif import solve_gurobi_multiobj

from .formulation import MILPInstance
from .adapter import MILPResultAdapterGurobi, MILPSequentialAdapter
from .create_instance import create_instance
from .helpers import sanity_check_milp_gurobi

from .heuristics_mipstart_validate import apply_mip_start_from_heuristic, validate_mip_start
from .solve_gurobi_sequential import solve_two_phase_max_accept_then_min_energy
# -------------------------
# Optional imports (never break package import)
# -------------------------
try:
    # If these exist, expose them; otherwise keep package importable.
    from .milp_two_phase import build_multi_slice_model_with_accept  # noqa: F401
except Exception:
    build_multi_slice_model_with_accept = None

try:
    # Some repos name the function differently; try a few common ones.
    from .milp_two_phase import solve_gurobi_two_phase_max_accept_then_min_energy  # noqa: F401
except Exception:
    solve_gurobi_two_phase_max_accept_then_min_energy = None

try:
    from .milp_two_phase import milp_two_phase  # noqa: F401
except Exception:
    milp_two_phase = None

# Optional: sequential solver utilities (import only if available)
try:
    from .solve_gurobi_sequential import (
        solve_gurobi_sequential,
        _status_to_str,
        _gurobi_status_to_str,
        _gurobi_objval_to_str,
        _gurobi_mipgap_to_str,
        solve_gurobi_max_accept,
    )
except ImportError:
    solve_gurobi_sequential = None
    _status_to_str = None
    _gurobi_status_to_str = None
    _gurobi_objval_to_str = None
    _gurobi_mipgap_to_str = None
    solve_gurobi_max_accept = None
