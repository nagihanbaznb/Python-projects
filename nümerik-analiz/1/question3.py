# Nagihan Baz - 171805024 - Question 3

MAX = 100

def luDecomposition(mat, n):

	lower = [[0 for x in range(n)]
			for y in range(n)]
	upper = [[0 for x in range(n)]
			for y in range(n)]

	for i in range(n):

		for k in range(i, n):

			sum = 0
			for j in range(i):
				sum += (lower[i][j] * upper[j][k])

			upper[i][k] = mat[i][k] - sum

		for k in range(i, n):
			if (i == k):
				lower[i][i] = 1
			else:

				sum = 0
				for j in range(i):
					sum += (lower[k][j] * upper[j][i])

				lower[k][i] = int((mat[k][i] - sum) /
								upper[i][i])

	print("Lower Triangular\t\tUpper Triangular")

	for i in range(n):

		for j in range(n):
			print(lower[i][j], end="\t")
		print("", end="\t")

		for j in range(n):
			print(upper[i][j], end="\t")
		print("")



mat = [[3, 1, 2],
	[6, 3, 4],
	[3, 1, 5]]

luDecomposition(mat, 3)

