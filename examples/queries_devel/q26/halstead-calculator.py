#!/usr/bin/env python
import os
import sys
import math

# Replace following globals with appropriate values for any language
# Julia Operators taken from https://github.com/JuliaLang/julia/commit/71c0aa3e5660258af5c042058d5d8d3b82d93efb

operators=["+","-","*","\\","^","\"","\'",".","&lt;","&gt;","~","&amp;","|","[","]","(",")",";",":","%",",","!"]
keywords=["function","global","for","end","while","if","else","elseif","break","switch","case","otherwise","try","catch","end","const","immutable","import","importall","export","type","typealias","return","true","false","macro","quote","in","abstract","module","using","continue","do","join","aggregate","println","hpat","@acc"]

singleline_comment_op="#"
multiline_comment_start_op="#="
multiline_comment_end_op="=#"
n1= {}
n2={}
def filter_token(token):
    tok = token
    while tok:
        tok = break_token(tok)

def break_token(token):
    op_pos = len(token)
    for op in operators:
        if token.startswith(op):
            if op not in n1:
                n1[op]=1
            else:
                n1[op]+=1
            return token[len(op):]
        if op in token:
            op_pos = min(op_pos,token.find(op))

    remaing_token = token[:op_pos]
    for keyword in keywords:
        if remaing_token == keyword:
            if keyword not in n1:
                n1[keyword]=1
            else:
                n1[keyword]+=1

    if remaing_token not in n2:
        n2[remaing_token]=1
    else:
        n2[remaing_token]+=1

    return token[op_pos:]

def measure_halstead(N1,N2,n1,n2):
    Vocabulary = n1 + n2
    Volume = ( N1 + N2 ) * math.log(Vocabulary,2)
    Difficulty = ((n1 / 2) * (N2 / n2))
    Effort = Difficulty * Volume
    print(Effort)

def filter_comments(sourcecode_file):
    singleline_comment_op_pos=-1
    multiline_comment_start_op_pos=-1
    multiline_comment_end_op_pos=-1
    filtered_lines = []
    inside_comment=False
    with open(sourcecode_file,'r') as fil:
        for line in fil:
            if not line.strip():
                continue
            if singleline_comment_op in line:
                singleline_comment_op_pos = line.find(singleline_comment_op)
            if multiline_comment_start_op in line:
                multiline_comment_start_op_pos = line.find(multiline_comment_start_op)
            if multiline_comment_end_op in line:
                multiline_comment_end_op_pos = line.find(multiline_comment_end_op)

            if (not inside_comment and singleline_comment_op_pos != -1):
                filtered_lines.append(line[:singleline_comment_op_pos])
            elif (inside_comment and multiline_comment_end_op_pos != -1):
                inside_comment = False
            elif (multiline_comment_start_op_pos != -1):
                inside_comment = True
            elif (inside_comment):
                inside_comment = True
            else:
                filtered_lines.append(line)
            singleline_comment_op_pos=-1
            multiline_comment_start_op_pos=-1
            multiline_comment_end_op_pos=-1

    return filtered_lines

def main(sourcecode_file):
    lines = filter_comments(sourcecode_file)
    for line in lines:
        tokens = line.strip().split()
        for token in tokens:
            filter_token(token)

    for key,value in n1.items():
        print(key + " = " + str(value))
    measure_halstead(sum(n1.values()),sum(n2.values()),len(n1),len(n2))

if __name__ == "__main__":
    argn = len(sys.argv)
    if argn != 2:
        print("Usage: python <File Path for which Halstead Metric to calculate>")
        exit(1)
    main(sys.argv[1])
