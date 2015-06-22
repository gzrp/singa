#include <glog/logging.h>
#include <fcntl.h>
#include <fstream>
#include "utils/cluster.h"
#include "proto/cluster.pb.h"
#include <sys/stat.h>
#include <sys/types.h>
namespace singa {

std::shared_ptr<Cluster> Cluster::instance_;
Cluster::Cluster(const ClusterProto &cluster, int procs_id) {
  procs_id_=procs_id;
  cluster_ = cluster;
  SetupFolders(cluster);
  if(server_worker_separate())
    nprocs_=nworker_procs()+nserver_procs();
  else
    nprocs_=std::max(nworker_procs(), nserver_procs());
  CHECK_LT(procs_id, nprocs_);
  if(nprocs_>1){
    std::ifstream ifs(cluster.hostfile(), std::ifstream::in);
    std::string line;
    while(std::getline(ifs, line)&&endpoints_.size()<nprocs_){
      endpoints_.push_back(line);
    }
    CHECK_EQ(endpoints_.size(), nprocs_);
  }

  // locate the process id of every worker/server
  int ngrps=cluster_.nworker_groups(), grp_size=cluster_.nworkers_per_group();
  int procs;
  for(int i=0;i<ngrps;i++){
    for(int j=0;j<grp_size;j++){
      procs=(i*grp_size+j) / cluster_.nworkers_per_procs();
      procs_ids_[Hash(i,j,kWorkerLayer)]=procs;
      procs_ids_[Hash(i,j,kWorkerParam)]=procs;
    }
  }
  ngrps=cluster_.nserver_groups(), grp_size=cluster_.nservers_per_group();
  int offset=cluster_.server_worker_separate()? procs:0;
  for(int i=0;i<ngrps;i++){
    for(int j=0;j<grp_size;j++){
      procs_ids_[Hash(i,j,kServer)]=(i*grp_size+j) / cluster_.nservers_per_procs()+offset;
    }
  }

  auto rt=new ZKClusterRT(cluster_.zookeeper_host());
  rt->Init();
  cluster_rt_=shared_ptr<ClusterRuntime>(static_cast<ClusterRuntime*>(rt));
}

void Cluster::SetupFolders(const ClusterProto &cluster){
  // create visulization folder
  mkdir(vis_folder().c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

shared_ptr<Cluster> Cluster::Get(const ClusterProto& cluster, int procs_id){
  instance_.reset(new Cluster(cluster, procs_id));
  return instance_;
}

shared_ptr<Cluster> Cluster::Get() {
  if(!instance_) {
    LOG(ERROR)<<"The first call to Get should "
              <<"provide the sys/model conf path";
  }
  return instance_;
}
int Cluster::Hash(int gid, int id, int flag){
  int ret=-1;
  if(flag==kServer){
    ret=(flag*cluster_.nserver_groups()+gid)*cluster_.nservers_per_group() + id;
  }else{
    ret=(flag*cluster_.nworker_groups()+gid)*cluster_.nworkers_per_group() + id;
  }
  return ret;
}
int Cluster::ProcsIDOf(int group_id, int id, int flag){
  return procs_ids_.at(Hash(group_id, id, flag));
}

}  // namespace singa
