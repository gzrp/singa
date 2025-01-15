import torch
import argparse
from algorithm.src.model import SAMS
from algorithm.src.model import MoEDNN
from algorithm.src.model.afn import AFN
from algorithm.src.model.armnet import ARMNet
from algorithm.src.model.baseline import MeanMoE
from algorithm.src.model.cin import CIN
from algorithm.src.model.dnn import DNN
from algorithm.src.model.nfm import NFM
from algorithm.src.model.sparsemax_verticalMoe import SparseMax_VerticalSAMS
# from src.model.verticalMoE import VerticalSAMS
# from src.model.verticalMoE_Plus import VerticalMoE_Predict_Sams
from algorithm.src.model.verticalMoE import VerticalSAMS

def initialize_model(args: argparse.Namespace):


    if args.net == "dnn":
        return DNN(args.nfield, args.nfeat, args.data_nemb, args.output_size ,args.moe_hid_layer_len,
                   # model specific args
                   args.dropout)


    
    if args.net == "afn":
        return AFN(args.nfield, args.nfeat, args.data_nemb, args.output_size, args.moe_hid_layer_len,
                   # model specific args
                   args.nhid, args.dropout)
    
    if args.net == "cin":
        return CIN(args.nfield, args.nfeat, args.data_nemb, args.output_size, 
                   args.nhid, args.dropout)
    
    
    if args.net == "armnet":
        return ARMNet(args.nfield, args.nfeat, args.data_nemb, args.output_size, args.moe_hid_layer_len,
                      # model specific args
                      args.nhead, args.nhid, 2, args.dropout)
                      # alpha = 2.0 -> sparseMax
    
    if args.net == "nfm":
        return NFM(args.nfield, args.nfeat, args.data_nemb, args.output_size, args.moe_hid_layer_len,
                   # model specific args
                   args.dropout)
    
    if args.net == "sparsemax_vertical_sams":
        return SparseMax_VerticalSAMS(args)
    
    
    if args.net == "meanMoE":
        return MeanMoE(args)
    
    if args.net == "sparseMoE":
        return VerticalSAMS(args)
    
    # if args.net == "sams":
    #     return SAMS(args)

    # if args.net == "moe_dnn":
    #     return MoEDNN(args.nfield, args.nfeat, args.data_nemb, args.moe_num_layers, args.moe_hid_layer_len, args.dropout, args.K)
 
