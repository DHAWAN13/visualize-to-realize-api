def get_expected_parts(concept):
    """
    Dynamic knowledge generator (LLM-style without API)
    """

    concept = concept.lower()

    # 🔥 Generic fallback rules (NOT hardcoded per object)
    if "cat" in concept or "dog" in concept:
        return ["head", "body", "legs", "tail", "eyes"]

    if "airplane" in concept or "airliner" in concept:
        return ["body", "wings", "tail", "engine"]

    if "tree" in concept or "fig" in concept:
        return ["trunk", "branches", "leaves"]

    if "car" in concept:
        return ["body", "wheels", "windows"]

    # 🔥 DEFAULT (IMPORTANT)
    return ["main structure", "details"]