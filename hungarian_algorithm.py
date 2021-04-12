import numpy as np

# to compare our algorithm with the one already implemented
from scipy.optimize import linear_sum_assignment


# step 1 function that takes as input the initial matrix m
# in this function we update the value of this adjacency matrix by subtracting the smallest entry for each row for all
# the other entries in that row
def step1(m):
    # iteration over all rows. shape() returns a tuple with the dimensions of the matrix, the firs element is the
    # number fo rows
    for i in range(m.shape[0]):
        # update of all the values in the row
        m[i, :] = m[i, :] - np.min(m[i, :])


# step 2 function that takes as input the initial matrix m
# in this function we update the value of this adjacency matrix by subtracting the smallest entry for each column
# for all the other entries in that column
def step2(m):
    # iteration over all the columns. like above
    for i in range(m.shape[1]):
        m[:, i] = m[:, i] - np.min(m[:, i])


# step 3 function has to compute the minimum number of rows and columns that we have to consider to cover all the
# zeros in the adjacency matrix. Input: matrix m. Output: number of lines to cover all zeros
# step 1: find a start assignment that covers as many "tasks" as possible
# step 2: mark all the rows with no assignments
# step 3: mark all (unmarked) columns having zeros in newly marked rows
# step 4: mark all rows having assignments in newly marked columns
# Repeat from 3 for all the new marked rows
# finally: select MARKED COLUMNS and UNMARKED ROWS
def step3(m):
    # store number of lines
    dim = m.shape[0]
    # initialize a list where to store the indexes of the assigned rows
    assigned = np.array([])
    # matrix of assignments, corresponding value (i, j) will have value 1 if corresponding row i is assigned to
    # corresponding row j
    assignments = np.zeros(m.shape, dtype=int)

    # define an assignment covering most task as possible
    # iterate over rows
    for i in range(0, dim):
        # iterate over columns
        for j in range(0, dim):
            # if we find a zero and the row and column has not been assigned the we can assign the row to the column
            # and we set the value of the corresponding set to zero
            if m[i, j] == 0 and np.sum(assignments[:, j]) == 0 and np.sum(assignments[i, :]) == 0:
                assignments[i, j] = 1
                # store the values of the indexes of the rows
                assigned = np.append(assigned, i)

    # use assigned array to initialize
    rows = np.linspace(0, dim - 1, dim).astype(int)
    # marked rows which are all the non-assigned rows
    marked_rows = np.setdiff1d(rows, assigned)
    # initialize numpy arrays
    new_marked_rows = marked_rows.copy()
    marked_cols = np.array([])

    # iterate until we have new marked rows, namely until length of new_marked_rows is bigger than zero
    while len(new_marked_rows) > 0:
        # initialize new array
        new_marked_cols = np.array([], dtype=int)
        # iterate over all the columns and mark all the rows that have a zero in that row
        for nr in new_marked_rows:
            # in zeros_cols we store the indexes of the cols and then we append them to the new_marked_cols array
            zeros_cols = np.argwhere(m[nr, :] == 0).reshape(-1)
            # discard all the cols already marked with setdiff1d
            new_marked_cols = np.append(new_marked_cols, np.setdiff1d(zeros_cols, marked_cols))
        # update value of the marked_cols array appending  the new marked columns
        marked_cols = np.append(marked_cols, new_marked_cols)
        # set new_marked_rows as an empty array
        new_marked_rows = np.array([], dtype=int)

        # iterate over the new marked rows and update the new_marked_rows by appending the indexes of the rows that have
        # an assignment in that column
        for nc in new_marked_cols:
            new_marked_rows = np.append(new_marked_rows, np.argwhere(assignments[:, nc] == 1).reshape(-1))
        marked_rows = np.unique(np.append(marked_rows, new_marked_rows))

    # returns the indexes of marked rows and unmarked columns
    return np.setdiff1d(rows, marked_rows).astype(int), np.unique(marked_cols)


# check if the number of lines drawn is equal to the number of rows or columns
# if this condition is true the we can find an optimal assignment for zeros and the algorithm ends
# otherwise the step 5 is required
# def ste4(m):


# In step 5 we find the smallest entry not covered by any line, we subtract it from any line that is
# not closed out and add it to any line that is closed out. By doing this we add one more zeros to the matrix.
# At the end go again to step 3.
# This function takes as input the adjacency matrix and the covered lines computed in the previous steps
def step5(m, covered_rows, covered_cols):
    # the function modifies the matrix by finding the minimum entry over the uncovered cells and subtracting this value
    # from each row that is not closed out.
    # Then add it to each columns that is closed out
    # initialize two arrays in which we store the values of the uncovered rows and columns
    uncovered_rows = np.setdiff1d(np.linspace(0, m.shape[0] - 1, m.shape[0]), covered_rows.astype(int))
    uncovered_cols = np.setdiff1d(np.linspace(0, m.shape[1] - 1, m.shape[1]), covered_cols.astype(int))
    # initialize a variable min_val as max over matrix m
    min_val = np.max(m)
    # iterate over the uncovered rows and cols
    for i in uncovered_rows.astype(int):
        for j in uncovered_cols.astype(int):
            # update value of the minimum
            if m[i, j] < min_val:
                min_val = m[i, j]
    # subtract min value from all uncovered rows iterating over uncovered_rows array
    for i in uncovered_rows.astype(int):
        m[i, :] -= min_val

    # update cols
    for j in covered_cols.astype(int):
        m[:, j] += min_val

    return m


# returns indexes of the zeros
def find_rows_single_zeros(matrix):
    for i in range(0, matrix.shape[0]):
        if np.sum(matrix[i, :] == 0) == 1:
            j = np.argwhere(matrix[i, :] == 0).reshape(-1)[0]
            return i, j
    # returns false if it does not find any rows with single zeros
    return False


# returns indexes of the zeros
def find_cols_single_zeros(matrix):
    for i in range(0, matrix.shape[1]):
        if np.sum(matrix[:, i] == 0) == 1:
            j = np.argwhere(matrix[:, i] == 0).reshape(-1)[0]
            return i, j
    # returns false if it does not find any cols with single zeros
    return False


# computes assignment for rows and columns with a single zero
def assignment_single_zero_lines(m, assignment):
    # find rows with a single zero
    val = find_rows_single_zeros(m)
    while val:
        i, j = val[0], val[1]
        # assign value 1 to zeros in the matrix
        m[i, j] += 1
        # discard all zeros in same column
        m[:, j] += 1
        # repeat procedure for columns
        assignment[i, j] = 1
        val = find_rows_single_zeros(m)

    val = find_cols_single_zeros(m)
    while val:
        i, j = val[0], val[1]
        m[i, :] += 1
        m[i, j] += 1
        assignment[i, j] = 1
        val = find_cols_single_zeros(m)

    # the function returns a partial assignment obtained by assigning all the rows and columns with a single zero
    return assignment


# returns indexes of the zeros covered by a line with more than one zero
def first_zero(m):
    return np.argwhere(m == 0)[0][0], np.argwhere(m == 0)[0][1]


# initializes a matrix in which we set the value m if there's an assignment for that cell
def final_assignment(initial_matrix, m):
    assignment = np.zeros(m.shape, dtype=int)
    # assign the value returned by assignment_single_zero_lines
    assignment = assignment_single_zero_lines(m, assignment)
    # until the matrix has a zero we look for rows and columns with at least two zeros
    while np.sum(m == 0) > 0:
        # assign value of indexes to i, j
        i, j = first_zero(m)
        # set to zero corresponding cell in assignment matrix
        assignment[i, j] = 1
        # discard all zero in same matrix or column
        m[i, :] += 1
        m[:, j] += 1
        # check if we're setting new rows with a single zero -> call again assignment_single_zero_lines
        assignment = assignment_single_zero_lines(m, assignment)

    # return a matrix in which we have only the values and weights of the optimal matching
    # and a boolean matrix indicating which value of the graph we've chosen
    return assignment * initial_matrix, assignment


# put together all the functions
def hungarian_algorithm(matrix):
    m = matrix.copy()
    step1(m)
    step2(m)
    # initialize a variable where we'll store the minimum number of lines required to cover all zeros
    n_lines = 0
    max_length = np.maximum(m.shape[0], m.shape[1])
    # until n_lines is not equal to max_length we call step3
    while n_lines != max_length:
        # lines contains the rows and columns we use to cover all the zeros
        lines = step3(m)
        # update n_lines summing length of arrays returned by step3
        n_lines = len(lines[0]) + len(lines[1])
        if n_lines != max_length:
            # pass matrix m and current lines used to cover all the zeros
            step5(m, lines[0], lines[1])
    # return final assignment passing initial matrix and computed matrix
    return final_assignment(matrix, m)


# initialize a random adjacency matrix with numpy
a = np.random.randint(100, size=(13, 13)) + 1

print(a)
res = hungarian_algorithm(a) #Minimum weights
print("\n Optimal Matching:\n", res[1], "\n Value: ", np.sum(res[0]))

maxElem = np.amax(a)
newres = hungarian_algorithm(np.vectorize(lambda x: maxElem - x)(a)) #Maximum weights
maskedarr= np.multiply(a,newres[1]) #Mask the original data with the hungarian result

print("\n Optimal Matching:\n", newres[1], "\n Value: ", np.sum(maskedarr))