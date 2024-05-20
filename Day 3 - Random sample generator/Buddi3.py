import numpy as np

def drawSamples(pmf: dict[str, float], n: int) -> list[str]:
    cmf=[]
    samples=[]
    keys=list(pmf.keys())
    s=0
    for i in pmf.values():
        s+=i
        cmf.append(s)
    for i in range(n):
        randomNum=(np.random.uniform(0,1))
        j = 0
        while randomNum > cmf[j]:
            j += 1
        samples.append(keys[j])
    return samples
        
pmf = {'Apple': 0.5, 'Banana': 0.3, 'Carrot': 0.2}
n = 10
print(drawSamples(pmf, n))