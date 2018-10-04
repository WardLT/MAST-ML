__init__ = ['score']

def score(x, models ,splitters):
    if x:
        # Get model
        name = x['estimator']
        x['estimator'] = models[name]
        del models[name]
        # Get cv
        name = x['cv']
        splitter_count = 0
        for splitter in splitters:
            if name in splitter:
                x['cv'] = splitter[1]
                break
            else:
                splitter_count += 1
        del splitters[splitter_count]

    return x, models, splitters
