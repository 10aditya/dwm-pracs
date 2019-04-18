from numpy import *
def hits(A):
    n=len(A)
    Au=dot(transpose(A),A)
    Hu=dot(A,transpose(A))
    a=ones(n)
    h=ones(n)
    print (a,h)
    for j in range(5):
        a=dot(a,Au)
        a=a/sum(a)
        h=dot(h,Hu)
        h=h/sum(h)
        print (a,h)
print("Here are the results of HITS algorithm with the small graph G1:")
A1 = [[0, 0, 1, 0], [0, 0, 1, 0],[0,0,0,1],[1,1,1,0]]
hits(A1)
print("Here are the results of HITS algorithm with the larger graph G2:")
A2 = [[0, 1, 0, 0,1, 0], [0, 0, 1, 0, 0, 0],[ 1,1,0, 1, 0,0],  [1,0,0,0,1,0], [1,0, 0,1, 0, 0],[0,1,1,0,0,0]]
hits(A2)

