#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:24:24 2023

@author: dennischang
"""
##############################################
# 1. longer length first or earliest deadline first
# 2. more number of hops first
# 3. FIFO
##############################################
                #[src,dst,len,prd,delay,]
flow_info = {
            "0": [5, 7, 1, 16,  738],
            "1": [0, 3, 2,  8,  724],
            "2": [11,2, 1,512,  583],
            "3": [ 7,9, 4,  4,  819],
            "4": [ 3,8, 1, 32,  978],
            "5": [ 9,0, 3,1024,1020],
            }
# flow_info.sort(key = lambda s: s[2])
sorted_flow_info = dict(sorted(flow_info.items(), key=lambda x: x[1][4],reverse=False)) 
print(sorted_flow_info)