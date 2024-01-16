package com.test.array;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

public class ArrayTest {

	public static void main(String[] args) {
		int[] cost = { 10, 15, 20 };
		int[] cost2 = { 1, 100, 1, 1, 1, 100, 1, 1, 100, 1 };
		int[] arr1 = { 2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19 }, arr2 = { 2, 1, 4, 3, 9, 6 };
		int[] nums = { 1, 3, 5, 6 };
		int[][] isConnected = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

		String[] words = { "cat", "bt", "hat", "tree" };
		String[] words2 = { "hello", "world", "leetcode" };

		ArrayTest test = new ArrayTest();

		System.out.println(test.countCharacters(words, "atach"));
		System.out.println(test.countCharacters(words2, "welldonehoneyr"));

		System.out.println(test.findCircleNum(isConnected));

		System.out.println(test.searchInsert(nums, 5));
		System.out.println(test.searchInsert(nums, 7));
		System.out.println(test.searchInsert(nums, 4));
		System.out.println(test.searchInsert(nums, 2));

		System.out.println(test.minCostClimbingStairs(cost));
		System.out.println(test.minCostClimbingStairs(cost2));

		System.out.println(Arrays.toString(test.relativeSortArray(arr1, arr2)));
		
	}

	// Min Cost Climbing Stairs
	/*
	 * You are given an integer array cost where cost[i] is the cost of ith step on
	 * a staircase. Once you pay the cost, you can either climb one or two steps.
	 * 
	 * You can either start from the step with index 0, or the step with index 1.
	 * 
	 * Return the minimum cost to reach the top of the floor.
	 */
	public int minCostClimbingStairs(int[] cost) {
		int len = cost.length;
		int[] dp = new int[len];
		dp[0] = cost[0];
		dp[1] = cost[1];
		for (int i = 2; i < len; i++) {
			dp[i] = cost[i] + Math.min(dp[i - 1], dp[i - 2]);
		}
		return Math.min(dp[len - 1], dp[len - 2]);
	}

	public int[] relativeSortArray(int[] arr1, int[] arr2) {
		int[] frequency = new int[1010];
		for (int i = 0; i < 1010; i++)
			frequency[i] = 0;
		for (int i = 0; i < arr1.length; i++)
			frequency[arr1[i]]++;
		int idx = 0;
		for (int i = 0; i < arr2.length; i++) {
			while (frequency[arr2[i]] > 0) {
				arr1[idx++] = arr2[i];
				frequency[arr2[i]]--;
			}
		}
		for (int i = 0; i < 1010; i++)
			while (frequency[i] > 0) {
				arr1[idx++] = i;
				frequency[i]--;
			}
		return arr1;
	}

	public int searchInsert(int[] nums, int target) {
		int start = 0;
		int end = nums.length - 1;
		while (start <= end) {
			int mid = start + (end - start) / 2;
			if (nums[mid] == target) {
				return mid;
			} else if (target < nums[mid]) {
				end = mid - 1;
			} else {
				start = mid + 1;
			}
		}
		return start;
	}

	// Depth First Search
	public int findCircleNum(int[][] M) {
		int[] visited = new int[M.length];
		int count = 0;
		for (int i = 0; i < M.length; i++) {
			if (visited[i] == 0) {
				dfs(M, visited, i);
				count++;
			}
		}
		return count;
	}

	public void dfs(int[][] M, int[] visited, int i) {
		for (int j = 0; j < M.length; j++) {
			if (M[i][j] == 1 && visited[j] == 0) {
				visited[j] = 1;
				dfs(M, visited, j);
			}
		}
	}

	public int countCharacters(String[] words, String chars) {
		HashMap<Character, Integer> countMap = new HashMap<>();
		for (char c : chars.toCharArray()) {
			countMap.put(c, countMap.getOrDefault(c, 0) + 1);
		}
		int res = 0;
		HashMap<Character, Integer> copyMap;
		for (String word : words) {
			copyMap = (HashMap<Character, Integer>) countMap.clone();
			boolean fail = false;
			for (char c : word.toCharArray()) {
				if (copyMap.get(c) == null || copyMap.get(c) <= 0) {
					fail = true;
					break;
				} else {
					copyMap.put(c, copyMap.get(c) - 1);
				}
			}
			if (!fail)
				res += word.length();
		}
		return res;
	}

	public static int sockMerchant(int n, List<Integer> ar) {
		HashMap<Integer, Integer> count = new HashMap<>();
		for (Integer color : ar) {
			if (count.containsKey(color)) {
				int newCount = count.get(color) + 1;
				count.put(color, newCount);
			} else {
				count.put(color, 1);
			}
		}
		int result = 0;
		for (Integer color : count.keySet()) {
			result += count.get(color) / 2;
		}
		return result;
	}

	ArrayList<Integer> arrange(ArrayList<Integer> A, ArrayList<Integer> B, int n) {
		Person[] people = new Person[n];
		for (int i = 0; i < n; i++) {
			people[i] = new Person(A.get(i), B.get(i));
		}
		Arrays.sort(people, new Person());

		Person[] rst = new Person[n];
		for (Person p : people) {
			int count = 0;
			for (int i = 0; i < n; i++) {
				if (count == p.infront) {
					while (rst[i] != null && i < n - 1) {
						i++;
					}
					rst[i] = p;
					break;
				}
				if (rst[i] == null)
					count++;
			}

		}
		ArrayList<Integer> heightrst = new ArrayList<Integer>();
		for (int i = 0; i < n; i++) {
			heightrst.add(rst[i].height);
		}
		return heightrst;
	}

	class Person implements Comparator<Person> {
		int height;
		int infront;

		public Person() {

		}

		public Person(int height, int infront) {
			this.height = height;
			this.infront = infront;
		}

		public int compare(Person p1, Person p2) {
			return p1.height - p2.height;
		}
	}

	
}
