from bkmedoids import BKmedoids
class Test:
    @staticmethod
    def test(ds, k, config, iterations):
        bics = []
        for it in range(iterations):
            config["seed"] = it
            bkmedoids = BKmedoids(ds,k,config)
            bkmedoids.run()
            bics.extend(bkmedoids.medoids)
        return bics
            