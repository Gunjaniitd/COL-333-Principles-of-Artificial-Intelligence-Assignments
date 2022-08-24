import pandas as pd
import numpy as np
import sys
import os
import json

#Some global variables

n = 0
d = 0
max_m = 0
max_a = 0
max_e = 0
max_r = 0
S = 0
T = 0

M = 1  #Morning 
A = 2  #Afternoon 
E = 3  #Evening 
R = 4  #Rest

grid = [[0 for i in range(d)] for j in range(n)]
last_holiday = [-1 for j in range(n)]

prev_M = []
prev_A = []
prev_E = []
prev_R = []

soln_list = []
count_m_e = []

def initialize():
    m_count, a_count, e_count = max_m, max_a, max_e

    if(d == 0):
        return

    for i in range(n):
        if(m_count != 0):
            grid[i][0] = M
            m_count -= 1
            prev_M.append(i)
        elif(e_count != 0):
            grid[i][0] = E
            e_count -= 1
            prev_E.append(i)
        elif(a_count != 0):
            grid[i][0] = A
            a_count -= 1
            prev_A.append(i)
        else:
            grid[i][0] = R
            prev_R.append(i)
            last_holiday[i] = 0

def sort_func(x):
    return last_holiday[x]

def sort_final(x):
    return count_m_e[x]

def part_a():
    initialize()

    global n, d, max_m, max_a, max_e, max_r
    global grid, prev_M, prev_A, prev_E, prev_R

    for j in range(1,d):
        m_count, a_count, e_count, r_count = max_m, max_a, max_e, max_r
        prev_A = sorted(prev_A, key=sort_func)

        # Morning shift

        if(n == 0):
            return False
        
        while(m_count != 0):
            if(len(prev_R) != 0):
                index = prev_R.pop()
            elif(len(prev_A) != 0):
                index = prev_A.pop()
            else:
                return False

            grid[index][j] = M
            m_count -= 1

        left = prev_M + prev_A + prev_E + prev_R
        left = (sorted(left, key=sort_func))
        left.reverse()

        #Rest day

        while(r_count != 0):
            if(len(left) != 0):
                index = left.pop()
            else:
                return False

            grid[index][j] = R
            last_holiday[index] = j
            r_count -= 1

        #Afternoon and evening shift

        while(e_count != 0):
            if(len(left) != 0):
                index = left.pop()
            else:
                return False

            grid[index][j] = E
            e_count -= 1

        while(a_count != 0):
            if(len(left) != 0):
                index = left.pop()
            else:
                return False

            grid[index][j] = A
            a_count -= 1
        
        prev_M = [] 
        prev_A = [] 
        prev_E = []
        prev_R = []

        for i in range(n):
            if(grid[i][j] == M):
                prev_M.append(i)
            elif(grid[i][j] == A):
                prev_A.append(i)
            elif(grid[i][j] == E):
                prev_E.append(i)
            else:
                prev_R.append(i)

    # print(np.matrix(grid))
                
    return True

def part_b():
    return part_a()
    
def check():
    global n, d, max_m, max_a, max_e, max_r, grid

    #checking morning constraint

    for i in range(n):
        for j in range(d):
            if(j != 0 and grid[i][j] == M):
                if(grid[i][j-1] == M or grid[i][j-1] == E):
                    return False

    #checking rest

    num_W = d//7
    for i in range(n):
        for j in range(num_W):
            check_flag = False 
            for k in range(j*7, (j+1)*7):
                if(grid[i][k] == R):
                    check_flag = True
            if(not check_flag):
                return False

    #checking number of nurses

    for j in range(d):
        c_m = max_m
        c_a = max_a
        c_e = max_e
        c_r = max_r

        for i in range(n):
            if(grid[i][j] == M):
                c_m -=1
            elif(grid[i][j] == A):
                c_a -= 1
            elif(grid[i][j] == E):
                c_e -= 1
            else:
                c_r -= 1

        if(c_m != 0 or c_a != 0 or c_e != 0 or c_r != 0):
            return False
  
    return True

def main():
    filename = sys.argv[1]

    global n, d, max_m, max_a, max_e, max_r, grid, last_holiday, soln_list
    global prev_M, prev_A, prev_E, prev_R, S, T, count_m_e

    df = pd.read_csv(filename)

    if (len(df.columns) == 5):
        N = df['N']
        D = df['D']
        mat_m = df['m']
        mat_a = df['a']
        mat_e = df['e']
        
        for i in range(len(N)):
            n = N[i]
            d = D[i]
            max_m = mat_m[i]
            max_a = mat_a[i]
            max_e = mat_e[i]
            max_r = n - (max_m + max_a + max_e)

            grid = [[0 for i in range(d)] for j in range(n)]
            last_holiday = [-1 for j in range(n)]

            prev_M = []
            prev_A = []
            prev_E = []
            prev_R = []

            ans = part_a()

            # print(np.matrix(grid))

            dic = {}

            for j in range(n):
                for k in range(d):
                    temp = "N" + str(j) + "_" + str(k)

                    if (grid[j][k] == 1):
                        dic[temp] = "M"
                    elif (grid[j][k] == 2):
                        dic[temp] = "A"
                    elif (grid[j][k] == 3):
                        dic[temp] = "E"
                    else:
                        dic[temp] = "R"

            if (check() and ans):
                soln_list.append(dic)
            else:
                soln_list.append({})

    if (len(df.columns) == 7):
        N = df['N']
        D = df['D']
        mat_m = df['m']
        mat_a = df['a']
        mat_e = df['e']
        mat_s = df['S']
        mat_t = df['T']
        
        for i in range(len(N)):
            n = N[i]
            d = D[i]
            max_m = mat_m[i]
            max_a = mat_a[i]
            max_e = mat_e[i]
            max_r = n - (max_m + max_a + max_e)
            S = mat_s[i]
            T = mat_t[i]

            grid = [[0 for i in range(d)] for j in range(n)]
            last_holiday = [-1 for j in range(n)]

            prev_M = []
            prev_A = []
            prev_E = []
            prev_R = []

            ans = part_b()

            dic = {}
            count_m_e = [0 for i in range(n)]

            for j in range(n):
                for k in range(d):
                    if (grid[j][k] == 1):
                        count_m_e[j] += 1
                    elif (grid[j][k] == 3):
                        count_m_e[j] += 1
            
            order = [l for l in range(n)]
            order = sorted(order, key=sort_final)
            order.reverse()
            new_grid = []
            for l in range(n):
                new_grid.append(grid[order[l]])

            for j in range(n):
                for k in range(d):
                    temp = "N" + str(j) + "_" + str(k)

                    if (new_grid[j][k] == 1):
                        dic[temp] = "M"
                    elif (new_grid[j][k] == 2):
                        dic[temp] = "A"
                    elif (new_grid[j][k] == 3):
                        dic[temp] = "E"
                    else:
                        dic[temp] = "R"

            senior_count = 0
            count_m_e.sort()            
            for k in range(S):
                senior_count += count_m_e[n-k-1]

            # print(np.matrix(new_grid))
            # print("Score: " + str(senior_count))    

            if (check() and ans):
                soln_list.append(dic)
            else:
                soln_list.append({})

    with open("solution.json", "w") as file:
        for d in soln_list:
            json.dump(d,file)
            file.write("\n")

main()



