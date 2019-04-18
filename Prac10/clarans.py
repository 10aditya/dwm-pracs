from pyclustering.cluster.clarans import clarans
from pyclustering.utils import read_sample;
from pyclustering.utils import draw_clusters;
from pyclustering.utils import timedcall;

def cluster(number_clusters, iterations, maxneighbours):
    data = read_sample('data.data')
    m_clarans = clarans(data, number_clusters, iterations, maxneighbours)
    (ticks, result) = timedcall(m_clarans.process)
    print("Execution time: ", ticks, "\n")
    clusters = m_clarans.get_clusters()
    draw_clusters(data, clusters)


#with 3 clusters
cluster(3,10,3)

#with 5 clusters
cluster(5,10,4)
