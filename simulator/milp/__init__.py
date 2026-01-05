from .topology_to_instance import build_instance_from_topology
from .solver_gurobi import solve_gurobi
from .solver_gurobi_multiobjectif import solve_gurobi_multiobj

# New shrink-until-feasible solver
#from .solve_gurobi_shrink_until_feasible import solve_gurobi_shrink_until_feasible

from .formulation import MILPInstance
from .adapter import MILPResultAdapterGurobi, MILPSequentialAdapter
from .create_instance import create_instance
from .helpers import sanity_check_milp_gurobi

# Optional: sequential solver utilities (import only if available)
try:
    from .solve_gurobi_sequential import (
        solve_gurobi_sequential,
        _status_to_str,
        _gurobi_status_to_str,
        _gurobi_objval_to_str,
        _gurobi_mipgap_to_str,
        solve_gurobi_shrink_until_feasible
    )
except ImportError:
    # Keep package importable even if sequential module symbols are missing
    solve_gurobi_sequential = None
    _status_to_str = None
    _gurobi_status_to_str = None
    _gurobi_objval_to_str = None
    _gurobi_mipgap_to_str = None
