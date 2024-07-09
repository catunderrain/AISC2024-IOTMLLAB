
def Dirty():
    Q =list('bach khoa'); C ="bcdfghjklmnpqrstvwxz"; V ="aeiouy"; Cr = {}; Vr = {}
    for i in range(len(Q)):
        for tx in range(len(Q)-i):
            if Q[tx] in C: re = ''.join(Q[tx:tx+i+1]).upper(); Cr[re] = Cr[re] + 1 if re in Cr else 1
            if Q[tx] in V: re = ''.join(Q[tx:tx+i+1]).upper(); Vr[re] = Vr[re] + 1 if re in Vr else 1
    for num, (re, i) in enumerate(Cr.items()): print(num, re, i)
    for num, (re, i) in enumerate(Vr.items()): print(num, re, i)
    print(f'A score: {sum(Cr.values())}\nB score: {sum(Vr.values())}')
    

def Clean():
    ques =list(input('Your string: '))
    print('-'*100)
    con ="bcdfghjklmnpqrstvwxz"
    vow ="aeiouy"
    conre = {}
    vowre = {}
    for i in range(len(ques)):
        for tx in range(len(ques)-i):
            if ques[tx] in con:
                re = ''.join(ques[tx:tx+i+1]).upper()
                if re in conre:
                    conre[re] += 1
                else:
                    conre[re] = 1
            if ques[tx] in vow:
                re = ''.join(ques[tx:tx+i+1]).upper()
                if re in vowre:
                    vowre[re] += 1
                else:
                    vowre[re] = 1
    for num, (re, i) in enumerate(conre.items()):
        print(num, re, i)
    print('-'*100)
    for num, (re, i) in enumerate(vowre.items()):
        print(num, re, i)
    print('-'*100)
    print(f'A score: {sum(conre.values())}')
    print(f'B score: {sum(vowre.values())}')
    
    
if __name__ == '__main__':
    Dirty()
    Clean()