package com.test.hackerrank;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;

public class CastleOnTheGridSolution {

	static int minimumMoves(List<String> grid, int startX, int startY, int goalX, int goalY) {
		ArrayDeque<Position> queue = new ArrayDeque<Position>();
		int matrix[][] = new int[grid.size()][grid.size()];
		for (int i = 0; i < grid.size(); i++) {
			String curr = grid.get(i);
			for (int j = 0; j < curr.length(); j++) {
				matrix[i][j] = curr.charAt(j) == '.' ? 1 : -1;
			}
		}
		Position pos = new Position(startX, startY, 0);
		queue.offer(pos);
		while (!queue.isEmpty()) {
			Position current = queue.poll();
			if (current.row == goalX && current.col == goalY) {
				return current.number;
			}
			matrix[current.row][current.col] = 0;
			addMoves(queue, current, matrix);
		}
		return -1;
	}

	static void addMoves(Queue<Position> queue, Position current, int[][] matrix) {

		int row = current.row;
		int col = current.col;
		int number = current.number + 1;
		while (--row >= 0 && matrix[row][col] == 1) {
			queue.offer(new Position(row, col, number));
		}
		
		row = current.row;
		col = current.col;
		while (++row < matrix.length && matrix[row][col] == 1) {
			queue.offer(new Position(row, col, number));
		}

		row = current.row;
		col = current.col;
		while (--col >= 0 && matrix[row][col] == 1) {
			queue.offer(new Position(row, col, number));
		}

		row = current.row;
		col = current.col;
		while (++col < matrix.length && matrix[row][col] == 1) {
			queue.offer(new Position(row, col, number));
		}

	}

	public static void main(String[] args) {
		List<String> grid = Arrays.asList("...", ".X.", "...");
		int startX = 0, startY = 0, goalX = 1, goalY = 2;
		int result = minimumMoves(grid, startX, startY, goalX, goalY);
		System.out.println(result);

		List<String> grid2 = Arrays.asList(".X.", ".X.", "...");
		int startX2 = 0, startY2 = 0, goalX2 = 0, goalY2 = 2;
		int result2 = minimumMoves(grid2, startX2, startY2, goalX2, goalY2);
		System.out.println(result2);
	}

}

class Position {
	int row;
	int col;
	int number;

	public Position(int row, int col, int number) {
		this.row = row;
		this.col = col;
		this.number = number;
	}
}