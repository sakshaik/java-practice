package com.test.hackerrank;

import java.util.Arrays;
import java.util.List;
import java.util.Stack;

public class PoisonousPlantsSolution {

	public static void main(String[] args) {
		List<Integer> plants = Arrays.asList(3, 6, 2, 7, 5);
		System.out.println(PoisonousPlantsSolution.poisonousPlants(plants));
		plants = Arrays.asList(6, 5, 8, 4, 7, 10, 9);
		System.out.println(PoisonousPlantsSolution.poisonousPlants(plants));
	}

	static int poisonousPlants(List<Integer> p) {
		Stack<Pair> stack = new Stack<>();
		int cnt = 0;
		for (int i = p.size() - 1; i >= 0; i--) {
			int temp = 0;
			while (!stack.empty() && p.get(i) < stack.peek().val) {
				temp++;
				Pair pair = stack.pop();
				temp = Math.max(temp, pair.count);
			}
			cnt = Math.max(cnt, temp);
			stack.push(new Pair(p.get(i), temp));
		}

		return cnt;
	}

	static class Pair {
		int val, count;

		public Pair(int val, int count) {
			this.val = val;
			this.count = count;
		}
	}

}
