class ConfigurationSelector:

    def select(self, complexity):

        if complexity < 0.33:

            return "shallow"

        elif complexity < 0.66:

            return "medium"

        else:

            return "deep"