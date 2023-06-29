class Estimandizer:
    """
    Generate estimands expicitly.
    """

    def __init__(self, data_handler, estimands):
        self.data_handler = data_handler
        self.estimands = estimands

    def check_estimand(estimand):
        already_included = ["dem_votes", "gop_votes", "total_votes"]
        if estimand in already_included:
            return False
        return True
