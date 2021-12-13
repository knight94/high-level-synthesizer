# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 21:11:22 2021

@author: Naman Jain
"""

import graphviz as gviz
import copy as cp
import matplotlib.pyplot as plt

syntax_table = {"Declaration":['INPUT', 'OUTPUT', 'VARIABLE'],
                "Control-flow":['IF', 'ELSE', 'ENDIF', 'LOOP', 'ENDLOOP', 'EXIT', 'WAIT0', 'WAIT1'],
                "Computational":['MINUS', 'NOT', 'ASSIGN', 'ADD', 'SUB', 'MUL', 'DIV', 'REM', 'EQ', 'NE', 'GT', 'GE', 'LT', 'LE', 'AND', 'OR', 'XOR', 'INDEX', 'LEFT','RIGHT', 'CONCAT'],
                "end": ['END']}
cdfg_shape = {'BLOCK':'box', 'WAIT':'hexagon', 'MERGE':'invtriangle', 'BRANCH':'triangle', 'START':'oval'}

dfg_shape = {'OPERATION':'circle', 'READ':'box', 'WRITE':'box', 'CONST':'square', 'START':'oval', 'STOP':'oval'}

var_type = ["BIT", "BITVECTOR"]

def read_prg_file(filename):
    with open(filename, 'r') as tf:
        return tf.readlines()

def split_lines(prg_code):
    prg = dict()
    line = 1
    for l in prg_code:
        prg[line] = l.split()
        line += 1
    return prg

class DFG_node():
    """
    node_type : operation, select, read, write, const
    """
    def __init__(self, num, node_type, ops):
        self.id = num
        self.type = node_type
        self.ops = ops
        self.succ = []
        self.pred = []
        self.inpv = []
        self.outv = []
        
    
    def __str__(self):
        return "{}".format(self.ops)
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.id == other.id
    
    def add_succ(self, succ):
        if succ not in self.succ:
            self.succ.append(succ)
    
    def add_pred(self, pred):
        if pred not in self.pred:
            self.pred.append(pred)

def get_par_type(par, input_l, output_l, var_l):
    if par in input_l.keys():
        return input_l[par]
    elif par in output_l.keys():
        return output_l[par]
    elif par in var_l.keys():
        return var_l[par]
    else:
        return (None, None)
    
def check_V_def(V, d1, num, symbol_t, start, base, t):
    if V in ["'1'", "'0'", '1', '0']:
        c1 = DFG_node(base+str(num), 'CONST', V)
        num += 1
        start.add_succ(c1)
        c1.add_pred(start)
        c1.add_succ(d1)
        d1.add_pred(c1)
    elif V in symbol_t.keys():
        d1.add_pred(symbol_t[V][0])
        symbol_t[V][0].add_succ(d1)
        # symbol_t[V] = (d1, symbol_t[V][1])
    else:
        r1 = DFG_node(base+str(num), 'READ', 'READ '+V)
        num += 1
        start.add_succ(r1)
        r1.add_pred(start)
        r1.add_succ(d1)
        d1.add_pred(r1)
        symbol_t[V] = (r1, t)
    return num

def get_out_type(t1, t2, op):
    if op in ['ADD', 'SUB', 'REM', 'DIV']:
        return (t1[0], max(t1[1], t2[1]))
    elif op in ['EQ', 'NE', 'GT', 'GE', 'LT', 'LE', 'INDEX']:
        return ('BIT', 1)
    elif op in ['AND', 'OR', 'XOR', 'LEFT','RIGHT']:
        return (t1[0], t1[1])
    elif op in ['MUL', 'CONCAT']:
        return ('BITVECTOR', t1[1]+t2[1])

class CFG_node():
    """
    node_type : wait, merge, branch, block
    """
    def __init__(self, node_type, num, comp_state, input_l, output_l, var_l, check_var=None):
        self.id = num
        self.type = node_type
        self.succ = []
        self.pred = []
        self.symbol_t = dict()
        self.check_var = check_var
        if node_type == 'BLOCK':
            self.comp_state = cp.deepcopy(comp_state)
            self.dfg_node = self.block_to_dfg(input_l, output_l, var_l)
    
    def __str__(self):
        return "{}:{}".format(self.type, self.id)
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.id == other.id
    
    def add_succ(self, succ):
        if succ not in self.succ:
            self.succ.append(succ)
    
    def add_pred(self, pred):
        if pred not in self.pred:
            self.pred.append(pred)
            
    def block_to_dfg(self, input_l, output_l, var_l):
        self.symbol_t = dict()
        base = self.type+'@'+str(self.id)
        num = 0
        start = DFG_node(base+str(num), 'START', 'NOP')
        num += 1
        for n in self.comp_state:
            if n[0] in ['MINUS', 'NOT', 'ASSIGN']:
                d1 = DFG_node(base+str(num), 'OPERATION', n[0])
                num += 1
                t1 = get_par_type(n[2], input_l, output_l, var_l)
                if not t1:
                    print("ERR operand {} type not found".format(n[2]))
                d1.inpv.append(n[2])
                num = check_V_def(n[2], d1, num, self.symbol_t, start, base, t1)

                self.symbol_t[n[1]] = (d1, t1)
                d1.outv.append(n[1])
            else:
                d1 = DFG_node(base+str(num), 'OPERATION', n[0])
                num += 1
                t1 = get_par_type(n[2], input_l, output_l, var_l)
                if not t1:
                    print("ERR operand {} type not found".format(n[2]))

                t2 = get_par_type(n[3], input_l, output_l, var_l)
                if not t2:
                    print("ERR operand {} type not found".format(n[3]))
                num = check_V_def(n[2], d1, num, self.symbol_t, start, base, t1)
                num = check_V_def(n[3], d1, num, self.symbol_t, start, base, t2)
                d1.inpv.append(n[2])
                d1.inpv.append(n[3])

                self.symbol_t[n[1]] = (d1, get_out_type(t1, t2, n[0]))
                d1.outv.append(n[1])
        
        end = DFG_node(base+str(num), 'STOP', 'NOP')
        num += 1
        for k,v in self.symbol_t.items():
            if len(v[0].succ) == 0:
                w1 = DFG_node(base+str(num), 'WRITE', 'WRITE '+k)
                num += 1
                w1.add_pred(v[0])
                v[0].add_succ(w1)
                w1.add_succ(end)
                end.add_pred(w1)
        return start

def process_wait_statement(n1, split_prg, line, num, input_l, output_l, var_l):
    s = split_prg[line]
    if s[1] in input_l.keys() or s[1] in var_l.keys():
        w1 = CFG_node('WAIT', num, None, None, None, None, s[1])
        w1.add_pred(n1)
        n1.add_succ(w1)
        return w1, num+1, line+1
    else:
        print("ERR {} no such parameter declared\n")
        return None

def process_cond_statement(n1, split_prg, line, num, input_l, output_l, var_l):
    s = split_prg[line]
    if s[2] in input_l.keys() or s[2] in var_l.keys():
        w1 = CFG_node('BRANCH', num, None, None, None, None, s[2])
        w1.add_pred(n1)
        n1.add_succ(w1)
        num += 1
        line += 1
        s = split_prg[line]
        prev_node1 = w1
        while s[0] != 'ENDIF' and s[0] != 'ELSE':
            s1, num, line = parse_beh_construct(prev_node1, split_prg, line, num, input_l, output_l, var_l)
            s = split_prg[line]
            prev_node1 = s1
        
        flag_else = False
        if s[0] == 'ELSE':
            flag_else = True
            line += 1
            s = split_prg[line]
            prev_node2 = w1
            while s[0] != 'ENDIF':
                s2, num, line = parse_beh_construct(prev_node2, split_prg, line, num, input_l, output_l, var_l)
                s = split_prg[line]
                prev_node2 = s2
        
        s3 = CFG_node('MERGE', num, None, None, None, None)
        s3.add_pred(prev_node1)
        prev_node1.add_succ(s3)
        if flag_else:
            s3.add_pred(prev_node2)
            prev_node2.add_succ(s3)
        else:
            s3.add_pred(w1)
            w1.add_succ(s3)
        return s3, num+1, line+1
    else:
        print("ERR {} parameter not declared".format(s[2]))
        return None

def process_loop_statement(n1, split_prg, line, num, input_l, output_l, var_l):
    w1 = CFG_node('MERGE', num, None, None, None, None)
    w1.add_pred(n1)
    n1.add_succ(w1)
    exit_nodes = list()
    num += 1
    line += 1
    s = split_prg[line]
    prev_node3 = w1
    while s[0] != 'ENDLOOP':
        if s[0] == 'EXIT':
            if s[2] in input_l.keys() or s[2] in var_l.keys():
                e1 = CFG_node('BRANCH', num, None, None, None, None, s[2])
                e1.add_pred(prev_node3)
                num += 1
                line += 1
                prev_node3.add_succ(e1)
                exit_nodes.append(e1)
                prev_node3 = e1
                s = split_prg[line]
                prev_node3 = e1
            else:
                print("ERR {} parameter not declared".format(s[2]))
                return None
        else:
            s4, num, line = parse_beh_construct(prev_node3, split_prg, line, num, input_l, output_l, var_l)
            s = split_prg[line]
            prev_node3 = s4
    
    prev_node3.add_succ(w1)
    w1.add_pred(prev_node3)
    if len(exit_nodes) > 1:
        s5 = CFG_node('MERGE', num, None, None, None, None)
        num += 1
        for e in exit_nodes:
            s5.add_pred(e)
            e.add_succ(s5)
        return s5, num, line+1
    elif len(exit_nodes) == 1:
        return exit_nodes[0], num, line+1
    
    print("WARN inifite loop detected")
    return prev_node3, num, line+1

def process_comp_statement(n1, split_prg, line, num, input_l, output_l, var_l):
    comp_state = []
    s = split_prg[line]
    while s[0] in syntax_table['Computational']:
        # print(s)
        if len(s) == 3:
            if s[2] not in input_l.keys() and s[2] not in output_l.keys() and s[2] not in var_l.keys() and s[2] not in ["'1'", "'0'", '1', '0']:
                print("ERR {} not declared\n".format(s[2]))
                return None
            if s[1] not in output_l.keys() and s[1] not in var_l.keys():
                var_l[s[1]] = ('extra', 0)
        elif len(s) == 4:
            if s[2] not in input_l.keys() and s[2] not in output_l.keys() and s[2] not in var_l.keys() and s[2] not in ["'1'", "'0'", '1', '0']:
                print("ERR {} not declared\n".format(s[2]))
                return None
            if s[3] not in input_l.keys() and s[3] not in output_l.keys() and s[3] not in var_l.keys() and s[3] not in ["'1'", "'0'", '1', '0']:
                print("ERR {} not declared\n".format(s[3]))
                return None
            if s[1] not in output_l.keys() and s[1] not in var_l.keys():
                var_l[s[1]] = ('extra', 0)
        else:
            print("ERR incomplete list of operands for {}".format(s[0]))
            return None
        
        comp_state.append(s)
        line += 1
        s = split_prg[line]
    
    b1 = CFG_node('BLOCK', num, comp_state, input_l, output_l, var_l)
    b1.add_pred(n1)
    n1.add_succ(b1)
    num += 1
    
    return b1, num, line

def parse_beh_construct(n1, split_prg, line, num, input_l, output_l, var_l):
    s = split_prg[line]
    # print(s)

    if s[0] == 'WAIT0' or s[0] == 'WAIT1':
        """
        Processing WAIT statements
        """
        return process_wait_statement(n1, split_prg, line, num, input_l, output_l, var_l)
    elif s[0] == 'IF':
        """
        Processing conditional statements
        """
        return process_cond_statement(n1, split_prg, line, num, input_l, output_l, var_l)
    elif s[0] == 'LOOP':
        """
        Processing loop statements
        """
        return process_loop_statement(n1, split_prg, line, num, input_l, output_l, var_l)
    elif s[0] in syntax_table['Computational']:
        """
        Processing computational statements
        """
        return process_comp_statement(n1, split_prg, line, num, input_l, output_l, var_l)

def graph_transversal(node, visited, dot, shape_d, block_list, expand=False, print_node=False):
    if node and node not in visited:
        visited.append(node)
        if expand:
            if node.type != 'BLOCK':
                if print_node:
                    print("{}\n".format(node))
                dot.node(str(node.id), str(node), shape=shape_d[node.type])
        else:
            if print_node:
                print("{}\n".format(node))
            dot.node(str(node.id), str(node), shape=shape_d[node.type])
        for ss in node.succ:
            if ss.type == 'BLOCK' and expand:
                block_list[str(ss)] = (ss.dfg_node, ss.symbol_t)
                with dot.subgraph(name='cluster'+str(ss.id)) as sb:
                    sb.attr(label=str(ss))
                    if print_node:
                        print("BLOCK DFG START")
                    graph_transversal(ss.dfg_node, [], sb, dfg_shape, {}, False, print_node)
                    if print_node:
                        print("BLOCK DFG END")
                    dot.edge(str(node.id), str(ss.dfg_node.id), lhead='cluster'+str(ss.id))
            elif node.type == 'BLOCK' and expand:
                dot.edge(str(node.dfg_node.id), str(ss.id), ltail='cluster'+str(node.id))
            else:
                dot.edge(str(node.id), str(ss.id))
            graph_transversal(ss, visited, dot, shape_d, block_list, expand, print_node)
    return None

def graph_node_edge(node, visited, E, P, S):
    if node and node not in visited:
        visited.append(node)
        S[node.id] = len(node.succ)
        for ss in node.succ:
            E.append((node.id, ss.id))
            if ss.id not in P.keys():
                P[ss.id] = 1
            else:
                P[ss.id] += 1
            graph_node_edge(ss, visited, E, P, S)
            
def get_last_node(node):
    if node.succ:
        for ss in node.succ:
            return get_last_node(ss)
    else:
            return node

# def get_asap_sch(dfgnode):
#     V = []
#     E = []
#     S = dict()
#     P = dict()
#     #assuming delay for operation = 1
#     di = 1
#     graph_node_edge(dfgnode, V, E, P, S)
#     T = dict()
#     for i in V:
#         T[i.id] = 1
#     T[dfgnode.id] = 0
#     U = dfgnode.succ.copy()
#     while U:
#         vv = U.pop(0)
#         for vvj in vv.succ:
#             if T[vvj.id] < di + T[vv.id]:
#                 T[vvj.id] = T[vv.id] + di
#             P[vvj.id] -= 1
#             if P[vvj.id] == 0:
#                 U.append(vvj)
    
#     return T 

def get_res_key(RES, ops):
    for k in RES.keys():
        if ops in RES[k][0]:
            return k
    return 0

def get_alap_sch(dfgnode, fu_list, latency):
    V = []
    E = []
    S = dict()
    P = dict()
    #assuming delay for operation = 1
    # di = 1
    graph_node_edge(dfgnode, V, E, P, S)
    T = dict()
    for i in V:
        T[i.id] = latency
    last = get_last_node(dfgnode)
    T[last.id] = latency
    U = last.pred.copy()
    while U:
        vv = U.pop(0)
        for vvj in vv.pred:
            if vv.type in ['READ', 'WRITE', 'CONST']:
                T[vvj.id] = T[vv.id]
            else:
                di = fu_list[get_res_key(fu_list, vv.ops)][2]
                if T[vvj.id] > T[vv.id] - di:
                    T[vvj.id] = T[vv.id] - di
            S[vvj.id] -= 1
            if S[vvj.id] == 0:
                U.append(vvj)
    return T

#resource list dict of functional unit types, k:([set of ops], bit width, delay)
def get_list_sch(dfgnode, fu_list, A):
    RES = fu_list.copy()
    RES[0] = (['READ', 'WRITE', 'CONST'], 1, 0)
    nres = len(RES)
    #initializing U for all resources and b; l = 20
    L = 20
    U = {}
    b = {}
    for k in range(0, nres):
        for l in range(1,L+1):
            U[(l,k)] = []
            b[(l,k)] = 0
    for k in range(0, nres):
        for vv in dfgnode.succ:
            if vv.type in RES[k][0]:
                U[(1,k)].append(vv)
            elif vv.ops in RES[k][0]:
                U[(1,k)].append(vv)
    V = []
    E = []
    Su = dict()
    P = dict()
    l = 1
    graph_node_edge(dfgnode, V, E, P, Su)
    T = dict()
    for i in V:
        T[i.id] = (1, -1, i.ops, i.inpv, i.outv)
    T[dfgnode.id] = (0, -1, dfgnode.ops, dfgnode.inpv, dfgnode.outv)
    flag = nres
    while flag:
        flag = nres
        for k in range(0, nres):
            if U[(l,k)]:
                if (k == 0) and U[(l,k)]:
                    S = U[(l,k)]
                    for vv in S:
                        T[vv.id] = (0, k, vv.ops, vv.inpv, vv.outv)
                        for vvj in vv.succ:
                            P[vvj.id] -= 1
                            if P[vvj.id] == 0:
                                U[(T[vvj.id][0],get_res_key(RES, vvj.ops))].append(vvj)
                else:
                    S = U[(l,k)][0:A[k-1]-b[(l,k)]]
                    U[(l+1,k)] = U[(l,k)][A[k-1]-b[(l,k)]:]
                    for vv in S:
                        T[vv.id] = (l, k, vv.ops, vv.inpv, vv.outv)
                        for p in range(0, RES[k][2]):
                            b[(l+p,k)] = b[(l+p, k)] + 1
                        for vvj in vv.succ:
                            if l+RES[k][2] > T[vvj.id][0]:
                                T[vvj.id] = (l+RES[k][2], get_res_key(RES, vvj.ops), vvj.ops, vvj.inpv, vvj.outv)
                            P[vvj.id] -= 1
                            if P[vvj.id] == 0 and vvj.type not in RES[0][0]:
                                U[(T[vvj.id][0],get_res_key(RES, vvj.ops))].append(vvj)
                            #TODO: check the write nodes
                            # elif vvj.type in RES[0][0]:
                            #     T[vvj.id] = (l+RES[k][2], 0)
                                
                    l += 1
            else:
                flag -= 1
    return T 

def get_list_sch_min_res(dfgnode, fu_list, latency):
    A = [1 for i in range(0, len(fu_list))]
    T_alap = get_alap_sch(dfgnode, fu_list, latency)
    for b,t in T_alap.items():
        if t < 0:
            return None
    RES = fu_list.copy()
    RES[0] = (['READ', 'WRITE', 'CONST'], 1, 0)
    nres = len(RES)
    #initializing U for all resources and b; l = 20
    L = 20
    U = {}
    b = {}
    for k in range(0, nres):
        for l in range(1,L+1):
            U[(l,k)] = []
            b[(l,k)] = 0
    for k in range(0, nres):
        for vv in dfgnode.succ:
            if vv.type in RES[k][0]:
                U[(1,k)].append(vv)
            elif vv.ops in RES[k][0]:
                U[(1,k)].append(vv)
    V = []
    E = []
    Su = dict()
    P = dict()
    l = 1
    #assuming delay for operation = 1
    graph_node_edge(dfgnode, V, E, P, Su)
    T = dict()
    for i in V:
        T[i.id] = (1, -1, i.ops, i.inpv, i.outv)
    T[dfgnode.id] = (0, -1, dfgnode.ops, dfgnode.inpv, dfgnode.outv)
    flag = nres
    while flag:
        flag = nres
        for k in range(0, nres):
            if U[(l,k)]:
                if (k == 0) and U[(l,k)]:
                    S = U[(l,k)]
                    for vv in S:
                        T[vv.id] = (0, k, vv.ops, vv.inpv, vv.outv)
                        for vvj in vv.succ:
                            P[vvj.id] -= 1
                            if P[vvj.id] == 0:
                                U[(l,get_res_key(RES, vvj.ops))].append(vvj)
                else:
                    S = U[(l,k)]
                    S_l = dict()
                    for vv in S:
                        slack = T_alap[vv.id] - l
                        if slack not in S_l.keys() and slack != 0:
                            S_l[slack] = []
                        if slack == 0:
                            T[vv.id] = (l, k, vv.ops, vv.inpv, vv.outv)
                            if A[k-1] <= b[(l,k)]:
                                A[k-1] += 1
                            for p in range(0, RES[k][2]):
                                b[(l+p,k)] = b[(l+p, k)] + 1
                            for vvj in vv.succ:
                                P[vvj.id] -= 1
                                if P[vvj.id] == 0 and vvj.type not in RES[0][0]:
                                    U[(l+RES[k][2],get_res_key(RES, vvj.ops))].append(vvj)
                                elif vvj.type in RES[0][0]:
                                    T[vvj.id] = (l+RES[k][2], 0, vvj.ops, vvj.inpv, vvj.outv)
                        else:
                            S_l[slack].append(vv)
                    free = A[k-1] - b[(l,k)]
                    if free > 0 and S_l:
                        for i in range(0, free):
                            kk = min(S_l.keys())
                            if S_l[kk]:
                                vv = S_l[kk].pop(0)
                                T[vv.id] = (l, k, vv.ops, vv.inpv, vv.outv)
                                for p in range(0, RES[k][2]):
                                    b[(l+p,k)] = b[(l+p, k)] + 1
                                for vvj in vv.succ:
                                    P[vvj.id] -= 1
                                    if P[vvj.id] == 0 and vvj.type not in RES[0][0]:
                                        U[(l+RES[k][2],get_res_key(RES, vvj.ops))].append(vvj)
                                    elif vvj.type in RES[0][0]:
                                        T[vvj.id] = (l+RES[k][2], 0, vvj.ops, vvj.inpv, vvj.outv)
                    for slk in S_l.keys():  
                        for vv in S_l[slk]:
                            U[(l+1,k)].append(vv)
                    l += 1
            else:
                flag -= 1
    return T, A

class Graph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self._graph_dict = graph_dict

    def edges(self, vertice):
        """ returns a list of all the edges of a vertice"""
        return self._graph_dict[vertice]
        
    def all_vertices(self):
        """ returns the vertices of a graph as a set """
        return set(self._graph_dict.keys())

    def all_edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        edge = set(edge)
        vertex1, vertex2 = tuple(edge)
        for x, y in [(vertex1, vertex2), (vertex2, vertex1)]:
            if x in self._graph_dict:
                self._graph_dict[x].append(y)
            else:
                self._graph_dict[x] = [y]
    
    def del_edge(self, edge):
        edge = set(edge)
        vertex1, vertex2 = tuple(edge)
        self._graph_dict[vertex1] = list(filter(lambda a: a!= vertex2, self._graph_dict[vertex1]))
        self._graph_dict[vertex2] = list(filter(lambda a: a!= vertex1, self._graph_dict[vertex2]))
    
    def del_vertex(self, vertex):
        for k in self._graph_dict[vertex]:
            self.del_edge({k,vertex})
        del self._graph_dict[vertex]

    def __generate_edges(self):
        """ A static method generating the edges of the 
            graph "graph". Edges are represented as sets 
            with one (a loop back to the vertex) or two 
            vertices 
        """
        edges = []
        for vertex in self._graph_dict:
            for neighbour in self._graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges
    
    def __iter__(self):
        self._iter_obj = iter(self._graph_dict)
        return self._iter_obj
    
    def __next__(self):
        """ allows us to iterate over the vertices """
        return next(self._iter_obj)

    def __str__(self):
        res = "vertices: "
        for k in self._graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

def get_compatibility_graph(T, A):
    E = {}
    for a in range(0, len(A)):
        E[a+1] = Graph()
    for k in T.keys():
        if T[k][1] != 0 and T[k][1] != -1:
            E[T[k][1]].add_vertex(k)
            for j in T.keys():
                if j != k and T[j][1] == T[k][1] and abs(T[j][0] - T[k][0]) >= A[T[k][1]-1]:
                    E[T[k][1]].add_edge({k,j})
    return E

def clique_partition_algo(E):
    clique_g = {}
    cg = 1
    while E.all_vertices():
        E_vert = list(E.all_vertices())
        num_v = -1
        vi = 0
        vj = 0
        for i in range(0, len(E_vert)):
            for j in range(i+1, len(E_vert)):
                if {E_vert[i], E_vert[j]} in E.all_edges():
                    n_v = len(set(E.edges(E_vert[i])) & set(E.edges(E_vert[j])))
                    if (n_v > num_v):
                        num_v = n_v
                        vi = E_vert[i]
                        vj = E_vert[j]
        if vi == 0 or vj == 0:
            #This means no edges in the compatibility graph
            for v in E_vert:
                clique_g[cg] = [v]
                cg += 1
                E.del_vertex(v)
            break;
        clique_g[cg] = [vi, vj]
        for vl in E_vert:
            if vl != vi and vl != vj:
                if set({vl, vj}) not in E.all_edges():
                    E.del_edge({vl, vi})
        E.del_vertex(vj)
        while(E.edges(vi)):
            n_m = -1
            Vkk = vi
            for vk in set(E.edges(vi)):
                if vk != vi:
                    n_v = len(set(E.edges(vi)) & set(E.edges(vk)))
                    if (n_v > n_m):
                        n_m = n_v
                        Vkk = vk
            clique_g[cg].append(Vkk)
            E_vert = list(E.all_vertices())
            for vl in E_vert:
                if vl != vi and vl != Vkk:
                    if set({vl, Vkk}) not in E.all_edges():
                        E.del_edge({vl, vi})
            E.del_vertex(Vkk)
        E.del_vertex(vi)
        cg += 1
    
    return clique_g

def extract_interval_graph_dfg(k, sym_t, T, fu_list, input_par):
    I = dict()
    gl_vl = []
    temp_T = sorted(T[k].values(), key=lambda x: x[0])
    if temp_T[-1][1] > 0:
        latency = temp_T[-1][0] + fu_list[temp_T[-1][1]][2]
    else:
        latency = temp_T[-1][0]
    for s in sym_t.keys():
        if s not in input_par.keys():
            # print("-----------------------------------")
            # print("variable: {}, start_node:{}".format(s,block_list[k][1][s][0].id))
            tup = T[k][sym_t[s][0].id]
            start_t = -1
            emax = -1
            if tup[1] > 0:
                start_t = tup[0]+fu_list[tup[1]][2]
                emax = tup[0]+fu_list[tup[1]][2]
            else:
                # print("start time: {}".format(tup[0]))
                start_t = tup[0]
                emax = tup[0]
            for v in sym_t[s][0].succ:
                v.type
                v_tup = T[k][v.id]
                if v_tup[1] != 0 and v_tup[1] != -1:
                    if emax < v_tup[0] + fu_list[v_tup[1]][2]:
                        emax = v_tup[0] + fu_list[v_tup[1]][2]
                else:
                    if emax < v_tup[0]:
                        emax = v_tup[0]
            print("For variable {} == start time:{} and end time:{}".format(s, start_t, emax))
            gv = input("Is this a Global variable (Y/N):")
            if gv == 'Y':
                start_t = 0
                emax = latency
                gl_vl.append(s)
            I[s] = (start_t, emax)
            #print("-----------------------------------")
    return I, gl_vl

def left_edge_algo(I):
    sort_I = dict(sorted(I.items(), key=lambda x:x[1][0]))
    I_C = dict()
    for k in sort_I.keys():
        I_C[k] = -1
    c = 0
    while(any(val == -1 for val in I_C.values())):
        r = -1
        S = []
        L = []
        for k,val in sort_I.items():
            if(val[0] > r):
                L.append(k)
        while(L):
            fs = L[0]
            S.append(fs)
            r = sort_I[fs][1]
            del sort_I[fs]
            L = []
            for k,val in sort_I.items():
                if(val[0] > r):
                    L.append(k)
        c += 1
        for v in S:
            I_C[v] = c
    plt.figure()
    for n,m in dict(sorted(I.items(), key=lambda x:x[1][0])).items():
        plt.plot(m,(n,n), 'ro-', color='blue')
    plt.show()
    return I_C

class datapath_node():
    """
    node_type : FU, mux, reg
    """
    def __init__(self, num, node_type):
        self.id = num
        self.type = node_type
        self.input = {}
        self.output = {}
        self.ops = []  
    
    def __str__(self):
        return "{}".format(self.ops)
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.id == other.id
    
    def add_ops(self, ops):
        if ops not in self.ops:
            self.ops.append(ops)
            
    def add_input(self, inp, v):
        if inp not in self.input:
            self.input[inp] = []
        if v not in self.input[inp]:
            self.input[inp].append(v)
            
    def add_output(self, op, v):
        if op not in self.output:
            self.output[op] = []
        if v not in self.output[op]:
            self.output[op].append(v)
            
    def find_ops(self, ops):
        if ops in self.ops:
            return True
        return False
    
            
def getnode_reg(reg, ops):
    for r in reg:
        if reg[r].find_ops(ops):
            return r
    
    #Control will not reach here
    return None

def getnode_fu(ful, v):
    for f in ful:
        if v in ful[f]:
            return f
    
    #Control will not reach here
    return None

# def create_datapath(sym_t, T, fu_block_l, input_par, registers, FUs):
#     for s in sym_t.keys():
#         # print(s)
#         if s in input_par.keys():
#             for v in sym_t[s][0].succ:
#                 if v.type not in ['READ', 'WRITE']:
#                     fu_type = T[v.id][1]
#                     fu_num = getnode_fu(fu_block_l[fu_type], v.id)
#                     data_node = FUs[fu_type][fu_num]
#                     data_node.add_input((s, 0), v.id)
#                     data_node.add_ops(v.ops)
#         else:
#             tv = sym_t[s][0]
#             fu_type = T[tv.id][1]
#             if fu_type > 0:
#                 fu_num = getnode_fu(fu_block_l[fu_type], tv.id)
#                 data_node = FUs[fu_type][fu_num]
#                 #find the register bind node
#                 rn = getnode_reg(registers, s)
#                 reg_n = registers[rn]
#                 data_node.add_output((s, rn), tv.id)
#                 reg_n.add_input((str(fu_type)+str(fu_num), (fu_type, fu_num)), tv.id)
#                 data_node.add_ops(tv.ops)
#                 for v in sym_t[s][0].succ:
#                     if v.type not in ['READ', 'WRITE']:
#                         fu_type = T[v.id][1]
#                         fu_num = getnode_fu(fu_block_l[fu_type], v.id)
#                         data_node = FUs[fu_type][fu_num]
#                         #find the register bind node
#                         rn = getnode_reg(registers, s)
#                         reg_n = registers[rn]
#                         data_node.add_input((s, rn), v.id)
#                         reg_n.add_output((str(fu_type)+str(fu_num), (fu_type, fu_num)), v.id)
#                         data_node.add_ops(tv.ops)

def create_datapath(sym_t, T, fu_block_l, input_par, registers, FUs):
    tmp = dict(sorted(T.items(), key=lambda x: x[1][0]))
    T_sort = {k:v for k, v in tmp.items() if v[0] > 0 and v[1] > 0}
    
    for b in T_sort:
        fu_type = T_sort[b][1]
        fu_num = getnode_fu(fu_block_l[fu_type], b)
        data_node = FUs[fu_type][fu_num]
        data_node.add_ops(T_sort[b][2])
        
        for s in T_sort[b][3]:
            if s in input_par:
                data_node.add_input((s, 0), b)
            elif s not in ["'1'", "'0'", '1', '0']:
                rn = getnode_reg(registers, s)
                reg_n = registers[rn]
                data_node.add_input((s, rn), b)
                reg_n.add_output((str(fu_type)+str(fu_num), (fu_type, fu_num)), b)
                
        for s in T_sort[b][4]:
            rn = getnode_reg(registers, s)
            reg_n = registers[rn]
            data_node.add_output((s, rn), b)
            reg_n.add_input((str(fu_type)+str(fu_num), (fu_type, fu_num)), b)

def graph_datapath(registers, muxs, FUs, input_par):
    dot = gviz.Digraph(format='png')
    dot.attr(rankdir='TB', compound='true')

    for i in input_par:
        with dot.subgraph() as s:
            s.node(i, i, shape='box', rank='same')
 
    for r in registers:
        with dot.subgraph() as s:
            s.node('R'+str(r), 'Reg'+str(r), shape='box', rank='same')
        
    for m in muxs:
        dot.node('M'+str(m), 'Mux'+str(m), shape='invtrapezium')
        
    for f in FUs:
        for fn in FUs[f]:
            with dot.subgraph() as s:
                s.node('F'+str(f)+str(fn), 'FU'+str(f)+str(fn), shape='box', rank='same')
        
            
    #Adding connections
    for m in muxs:
        if muxs[m].ops[0] == 'reg':
            r = muxs[m].output[m][0]
            dot.edge('M'+str(m), 'R'+str(r))
            for o in muxs[m].input[m]:
                dot.edge('F'+str(o[0])+str(o[1]), 'M'+str(m))
        else:
            fm = muxs[m].output[m][0]
            dot.edge('M'+str(m), 'F'+str(fm[0])+str(fm[1]))
            for o in muxs[m].input[m]:
                if type(o) == int:
                    dot.edge('R'+str(o), 'M'+str(m))
                else:
                    dot.edge(o, 'M'+str(m))
                    
    for r in registers:
        tmp = registers[r].input
        if len(tmp) == 1:
            tf = list(tmp.keys())[0]
            dot.edge('F'+str(tf[1][0])+str(tf[1][1]), 'R'+str(r))
            
    for f in FUs:
        for fn in FUs[f]:
            tmp = FUs[f][fn].input
            if len(tmp) == 1:
                r = list(tmp.keys())[0]
                if r[1]:
                    dot.edge('R'+str(r[1]), 'F'+str(f)+str(fn))
                else:
                    dot.edge(r[0], 'F'+str(f)+str(fn))
    
    return dot

class fsm_node():
    """
    node_type : wait, branch, block_per_cycle
    """
    def __init__(self, num, node_type):
        self.id = num
        self.type = node_type
        self.succ = {}
        self.output = {}
    
    def __str__(self):
        return "{}".format(self.id)
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.id == other.id
            
    def add_succ(self, succ, cond):
        if cond not in self.succ:
            self.succ[cond] = succ
        else:
            print("cond already exist\n")

def get_mux(muxs, node):
    index = []
    for m in muxs:
        if node.type == muxs[m].ops[0] and str(muxs[m].output[muxs[m].id][0][0]) + str(muxs[m].output[muxs[m].id][0][1]) == node.id:
            index.append(m)
    
    return index

def get_mux_r(muxs, node):
    index = []
    for m in muxs:
        if node.type == muxs[m].ops[0] and muxs[m].output[muxs[m].id][0] == node.id:
            index.append(m)
    
    return index

def get_port_mux(muxs, index, v):
    m_ind = []
    for i in index:
        if v in muxs[i].input[muxs[i].id]:
            m_ind.append(i)

    return m_ind
    
def create_fsm_block(base, fsm, dot, cond, registers, input_par, T, FUs, fu_block_l, muxs):
    tmp = dict(sorted(T.items(), key=lambda x: x[1][0]))
    T_sort = {k:v for k, v in tmp.items() if v[0] > 0 and v[1] > 0}
    
    cur_cycle = 1
    num = 1
    prev = fsm_node(int(str(base)+str(num)), 'BLOCK'+str(cur_cycle))
    num += 1
    fsm.add_succ(prev, cond)
    dot.edge(str(fsm.id), str(prev.id), label=cond)
    for b in T_sort:
        if cur_cycle != T_sort[b][0]:
            cond = 'clock cycle = ' + str(T_sort[b][0])
            cur_cycle = T_sort[b][0]
            dot.node(str(prev.id), str(prev.output), shape='box')
            new = fsm_node(int(str(base)+str(num)), 'BLOCK'+str(cur_cycle))
            num += 1
            prev.add_succ(new, cond)
            dot.edge(str(prev.id), str(new.id), label=cond)
            prev = new
            
        #Determine the control signal values for datapath
        #Find the FU binding using fu_block_l
        fu_type = T_sort[b][1]
        fu_num = getnode_fu(fu_block_l[fu_type], b)
        data_node = FUs[fu_type][fu_num]
        
        #FU control signal
        prev.output['FU'+str(fu_type)+str(fu_num)] = T_sort[b][2]
        
        #FU mux control signal
        ind = get_mux(muxs, data_node)
        # print(T_sort[b][3])
        if ind:
            for i in T_sort[b][3]:
                if i in input_par:
                    mux_in = get_port_mux(muxs, ind, i)
                    prev.output['MUX'+str(mux_in[0])] = i
                elif i not in ["'1'", "'0'", '1', '0']:
                    rn = getnode_reg(registers, i)
                    mux_in = get_port_mux(muxs, ind, rn)
                    if len(mux_in) == 1:
                        prev.output['MUX'+str(mux_in[0])] = i
                    else:
                        for mi in mux_in:
                            if 'MUX'+str(mi) not in prev.output:
                                prev.output['MUX'+str(mi)] = i
        
        #output reg control signal
        rn = getnode_reg(registers, T_sort[b][4][0])
        ind = get_mux_r(muxs, registers[rn])
        if ind:
            prev.output['MUX'+str(ind[0])] = 'FU'+data_node.id
    dot.node(str(prev.id), str(prev.output), shape='box')
    
    return prev 

def create_fsm(node, visited, dot, fsm, cond, node_shape, registers, input_par, T, FUs, fu_block_l, muxs):
    if node and node not in visited:
        visited.append(node)
        if node.type == 'WAIT':
            w = fsm_node(node.id, 'WAIT')
            fsm.add_succ(w, cond)
            dot.node(str(w.id), str(w.type), shape=node_shape)
            if node.check_var not in input_par:
                r = 'reg'+ str(getnode_reg(registers, node.check_var))
            else:
                r = node.check_var
            for ss in node.succ:
                if ss.type != 'MERGE' and ss.type != 'BLOCK':
                    dot.edge(str(w.id), str(ss.id), label=r+'=1')
                create_fsm(ss, visited, dot, w, r+'=1', node_shape, registers, input_par, T, FUs, fu_block_l, muxs)
            dot.edge(str(w.id), str(w.id), label=r+'=0')
        elif node.type == 'BRANCH':
            b = fsm_node(node.id, 'BRANCH')
            fsm.add_succ(b, cond)
            dot.node(str(b.id), str(b.type), shape=node_shape)
            tmp = 1
            if node.check_var not in input_par:
                r = 'reg'+ str(getnode_reg(registers, node.check_var))
            else:
                r = node.check_var
            for ss in node.succ:
                if ss.type != 'MERGE' and ss.type != 'BLOCK':
                    dot.edge(str(b.id), str(ss.id), label=r+'='+str(tmp))
                create_fsm(ss, visited, dot, b, r+'='+str(tmp), node_shape, registers, input_par, T, FUs, fu_block_l, muxs)
                tmp = 0
        elif node.type == 'MERGE':
            for ss in node.succ:
                if ss.type != 'MERGE' and ss.type != 'BLOCK':
                    dot.edge(str(fsm.id), str(ss.id), label=cond)
                create_fsm(ss, visited, dot, fsm, cond, node_shape, registers, input_par, T, FUs, fu_block_l, muxs)
        elif node.type == 'BLOCK':
            b = create_fsm_block(node.id, fsm, dot, cond, registers, input_par, T[str(node)], FUs, fu_block_l, muxs)
            for ss in node.succ:
                if ss.type != 'MERGE':
                    dot.edge(str(b.id), str(ss.id), label='True')
                create_fsm(ss, visited, dot, b, 'True', node_shape, registers, input_par, T, FUs, fu_block_l, muxs)
    elif node in visited:
        if node.succ[0].type != 'MERGE':
            dot.edge(str(fsm.id), str(node.succ[0].id), label=cond)
        else:
            create_fsm(node.succ[0], visited, dot, fsm, cond, node_shape, registers, input_par, T, FUs, fu_block_l, muxs)

if __name__ =="__main__":
    # input_file_path = "C:/Users/nsahu/Documents/Semester-II/COL719/Lab_Assignment_2/"
    # input_file = "modulo_HDL"
    # input_file_path = input("Enter the HDL file path:")
    input_file = input("Enter the HDL filename:")
    option = input("Enter option for CDFG visualization (1)BLOCK expanded in DFG or (2)BLOCK not expanded:")
    # filename = input_file_path + input_file
    filename = input_file
    prg_code = read_prg_file(filename)
    split_prg = split_lines(prg_code)
    
    cfg_graph = []
    num = 0
    
    decl = True
    line = 1
    input_par = dict()
    output_par = dict()
    var_par = dict()
    while decl:
        s = split_prg[line]
        line += 1
        if (s[0] in syntax_table['Declaration']):
            if s[1] not in input_par.keys() and s[1] not in output_par.keys() and s[1] not in var_par.keys():
                if s[0] == 'INPUT':
                    if len(s) == 3:
                        input_par[s[1]] = (s[2], 1)
                    else:
                        input_par[s[1]] = (s[2], s[3])
                elif s[0] == 'OUTPUT':
                    if len(s) == 3:
                        output_par[s[1]] = (s[2], 1)
                    else:
                        output_par[s[1]] = (s[2], s[3])
                elif s[0] == 'VARIABLE':
                    if len(s) == 3:
                        var_par[s[1]] = (s[2], 1)
                    else:
                        var_par[s[1]] = (s[2], s[3])
            else:
                print("ERR: multiple declaration found for variable : {}".format(s[1]))
        else:
            decl = False
            line -= 1
    
    n1 = CFG_node('START', num, None, None, None, None)
    num += 1
    
    s = split_prg[line]
    prev = n1
    while s[0] not in syntax_table['end']:
        n2, num, line =  parse_beh_construct(prev, split_prg, line, num, input_par, output_par, var_par)
        prev = n2
        s = split_prg[line]
    
    prev.add_succ(n1)
    n1.add_pred(prev)
    block_list = {}
    if option == '1':
        dot = gviz.Digraph(format='png')
        dot.attr(rankdir='TB', compound='true')
        graph_transversal(n1, [], dot, cdfg_shape, block_list, expand=True, print_node=False)
        # dot.render(filename = input_file_path+input_file+'_cdfg')
        dot.render(filename = input_file+'_cdfg')
        print(dot)
        dot.view()
    else:
        dot2 = gviz.Digraph(format='png')
        dot2.attr(rankdir='TB')
        graph_transversal(n1, [], dot2, cdfg_shape, block_list, expand=False, print_node=False)
        # dot2.render(filename = input_file_path+input_file+'_cfg')
        dot.render(filename = input_file+'_cdfg')
        print(dot2)
        dot2.view()

    # n_dfg = n1.succ[0].succ[0].succ[0].succ[0].dfg_node
    # dot1 = gviz.Digraph(format='png')
    # dot1.attr(rankdir='TB')
    # graph_transversal(n_dfg, [], dot1, dfg_shape)
    # dot1.render(filename = input_file_path+'dfg')
    # dot1.view()
    
    """
    Scheduling stage
    """
    option_s = input("Enter option for type of schedule (1) minimum latency (2) minimum resource:")
    #sample functional library with k:([set of ops], bit width, delay)
    fu_list = {1:(['MINUS', 'NOT', 'ASSIGN', 'ADD', 'SUB', 'EQ', 'NE', 'GT', 'GE', 'LT', 'LE', 'INDEX','AND', 'OR', 'XOR', 'LEFT','RIGHT'], 8, 2), 2:(['MUL', 'CONCAT', 'REM', 'DIV'], 16, 2)}
    
    #minimum latency resource constraint scheduling
    if option_s == '1':
        #number of resources
        A = [2,1]
        T_list = {}
        for k in block_list.keys():
            T_list[k] = get_list_sch(block_list[k][0], fu_list, A)
    else:
        #minimum resource time constraint scheduling
        #Latency per block
        T_list_minr = {}
        num_r = {}
        for k in block_list.keys():
            latency = input("Enter option latency constraint for {} (if no latency then enter 0):".format(k))
            if latency == '0':
                A = [1,1]
                T_list_minr[k] = get_list_sch(block_list[k][0], fu_list, A)
                num_r[k] = A.copy()
            else:
                T_list_minr[k], num_r[k] = get_list_sch_min_res(block_list[k][0], fu_list, int(latency))
    
    """
    Resource and register binding stage
    """
    """
    Operation binding using clique grouping
    """
    res_binding = {}
    for k in block_list.keys():
        if option_s == '1':
            C_g = get_compatibility_graph(T_list[k], A)
            for gg in C_g:
                E = C_g[gg]
                res_binding[(k, gg)] = clique_partition_algo(E)
        else:
            C_g = get_compatibility_graph(T_list_minr[k], num_r[k])
            for gg in C_g:
                E = C_g[gg]
                res_binding[(k, gg)] = clique_partition_algo(E)
    
    """
    Register binding
    """
    """
    Extract variable information from DFGs symbol table
    Interval Graph will be created using tuples for each variable
    """
    reg_binding = {}
    gv_ls = set()
    for k in block_list.keys():
        if option_s == '1':
            I, g = extract_interval_graph_dfg(k, block_list[k][1], T_list, fu_list, input_par)
        else:
            I, g = extract_interval_graph_dfg(k, block_list[k][1], T_list_minr, fu_list, input_par)
        I_C = left_edge_algo(I)
        reg_binding[k] = I_C
        gv_ls = gv_ls.union(set(g))
        
    
    """
    Datapath construction using MUX for input and output ports of register & FUs
    """
    """
    Find the Union of registers and FUs
    """
    registers = {}
    FUs = {}
    muxs = {}
    #Create registers for global variables
    num = 1
    for r in gv_ls:
        element = datapath_node(num, 'reg')
        element.add_ops(r)
        registers[num] = element
        num += 1
    
    gid = num
    for k in reg_binding:
        d = reg_binding[k]
        for r in d:
            if r not in gv_ls:
                if d[r] < gid:
                    tmp = d[r]+gid-1
                else:
                    tmp = d[r]
                if tmp not in registers:
                    element = datapath_node(d[r], 'reg')
                    element.add_ops(r)
                    registers[tmp] = element
                else:
                    registers[tmp].add_ops(r)
                    
    # for r in registers:
    #     print(registers[r].ops)
    fu_block_l = {}
    for k in res_binding:
        d = res_binding[k]
        if k[1] not in FUs:
            FUs[k[1]] = {}
            fu_block_l[k[1]] = {}
        for r in d:
                if r not in FUs[k[1]]:
                    element = datapath_node(str(k[1])+str(r), 'FU')
                    FUs[k[1]][r] = element
                    fu_block_l[k[1]][r] = []
                    fu_block_l[k[1]][r] += d[r]
                else:
                    fu_block_l[k[1]][r] += d[r]
                    
    for k in block_list.keys():
        # print("---------------------------")
        # print(k)
        if option_s == '1':
            create_datapath(block_list[k][1], T_list[k], fu_block_l, input_par, registers, FUs)
        else:
            create_datapath(block_list[k][1], T_list_minr[k], fu_block_l, input_par, registers, FUs)
        # print("---------------------------")
    
    num = 1
    for r in registers:
        if len(registers[r].input) > 1:
            mu = datapath_node(num, 'mux')
            mu.add_ops('reg')
            mu.add_output(num, r)
            for f in registers[r].input:
                mu.add_input(num, f[1])
            muxs[num] = mu
            num += 1
            
    for f in FUs:
        for fn in FUs[f]:
            #Determine number of inputs per port
            if (len(FUs[f][fn].input) > 1):
                tmp = {}
                ports = 1
                for k,l in FUs[f][fn].input.items():
                    for b in l:
                        if b in tmp:
                            tmp[b].append(k)
                            if (len(tmp[b]) > ports):
                                ports = len(tmp[b])
                        else:
                            tmp[b] = [k]
                for p in range(0, ports):
                    mu = datapath_node(num, 'mux')
                    mu.add_ops('FU')
                    mu.add_output(num, (f, fn))
                    for t in tmp:
                        if p < len(tmp[t]):
                            tmp_r = tmp[t][p][1]
                            if tmp_r:
                                mu.add_input(num, tmp_r)
                            else:
                                mu.add_input(num, tmp[t][p][0])
                    muxs[num] = mu
                    num += 1
                    
    data_dot = graph_datapath(registers, muxs, FUs, input_par)
    data_dot.render(filename = input_file+'datapath')
    # print(dot)
    data_dot.view()
    
    """
    FSM contruction
    """
    #create init node
    fsm_dot = gviz.Digraph(format='png')
    fsm_s = fsm_node(n1.id, 'INIT')
    fsm_dot.node(str(fsm_s.id), str(fsm_s.type), shape='circle')
    fsm_dot.edge(str(fsm_s.id), str(n1.succ[0].id), label='True')
    if option_s == '1':
        create_fsm(n1.succ[0], [], fsm_dot, fsm_s, 'True', 'circle', registers, input_par, T_list, FUs, fu_block_l, muxs)
    else:
        create_fsm(n1.succ[0], [], fsm_dot, fsm_s, 'True', 'circle', registers, input_par, T_list_minr, FUs, fu_block_l, muxs)
    fsm_dot.render(filename = input_file+'_fsm')
    fsm_dot.view()
    
                    