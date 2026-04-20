"""Agent tools — functions the LLM can call to interact with the knowledge base."""

from tools.tools import (
    search_policy_docuument,
    list_available_documents,
    extract_coverage_limits,
    compare_policies,
    flag_anomolies,
    generate_summary
)

ALL_TOOLS = [
    search_policy_docuument,
    list_available_documents,
    extract_coverage_limits,
    compare_policies,
    flag_anomolies,
    generate_summary
]
