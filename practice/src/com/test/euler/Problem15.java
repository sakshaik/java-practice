package com.test.euler;

public class Problem15 {

	private static final int NUMBER = 20;
	private static final long[][] grid = new long[NUMBER + 1][NUMBER + 1];

	public static void main(String[] args) {
		grid[0][0] = 0;
		for (int i = 1; i <= NUMBER; i++) {
			grid[i][0] = 1;
			grid[0][i] = 1;
		}
		for (int i = 1; i <= NUMBER; i++) {
			for (int j = 1; j <= i; j++) {
				grid[i][j] = grid[j][i] = grid[j - 1][i] + grid[j][i - 1];
			}
		}
		
		System.out.println(grid[NUMBER][NUMBER]);
	}
}
