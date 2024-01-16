package com.test.leetcode;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.stream.Stream;

public class LeetCodeTest {

	// Letter Combinations of a Phone Number
	public List<String> letterCombinations(String digits) {
		List<String> ans = new LinkedList<String>();
		if (digits.isEmpty())
			return ans;
		String[] mapping = new String[] { "0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
		ans.add(0, "");
		for (int i = 0; i < digits.length(); i++) {
			int x = Character.getNumericValue(digits.charAt(i));
			int size = ans.size();
			for (int k = 1; k <= size; k++) {
				String t = ans.remove(0);
				for (char s : mapping[x].toCharArray())
					ans.add(t + s);
			}
		}
		return ans;
	}

	// Index of Sum of two values in an array equal to target
	public int[] twoSum(int[] nums, int target) {
		int[] result = new int[2];
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < nums.length; i++) {
			if (map.containsKey(target - nums[i])) {
				result[1] = i;
				result[0] = map.get(target - nums[i]);
				return result;
			}
			map.put(nums[i], i);
		}
		return result;
	}

	// Add two numbers stored in reverse order in Linked List
	public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
		int carry = 0;
		ListNode p, dummy = new ListNode(0);
		p = dummy;
		while (l1 != null || l2 != null || carry != 0) {
			if (l1 != null) {
				carry += l1.val;
				l1 = l1.next;
			}
			if (l2 != null) {
				carry += l2.val;
				l2 = l2.next;
			}
			p.next = new ListNode(carry % 10);
			carry /= 10;
			p = p.next;
		}
		return dummy.next;
	}

	// Longest Substring Without Repeating Characters
	public int lengthOfLongestSubstring(String s) {
		if (s.length() == 0)
			return 0;
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		int max = 0;
		for (int i = 0, j = 0; i < s.length(); ++i) {
			if (map.containsKey(s.charAt(i))) {
				j = Math.max(j, map.get(s.charAt(i)) + 1);
			}
			map.put(s.charAt(i), i);
			max = Math.max(max, i - j + 1);
		}
		return max;
	}

	// Find median of two sorted arrays
	public double findMedianSortedArrays(int[] nums1, int[] nums2) {
		if (nums1.length > nums2.length)
			return findMedianSortedArrays(nums2, nums1);
		int x = nums1.length;
		int y = nums2.length;
		int low = 0;
		int high = x;
		while (low <= high) {
			int partX = (low + high) / 2;
			int partY = (x + y + 1) / 2 - partX;
			int xLeft = partX == 0 ? Integer.MIN_VALUE : nums1[partX - 1];
			int xRight = partX == x ? Integer.MAX_VALUE : nums1[partX];
			int yLeft = partY == 0 ? Integer.MIN_VALUE : nums2[partY - 1];
			int yRight = partY == y ? Integer.MAX_VALUE : nums2[partY];
			if (xLeft <= yRight && yLeft <= xRight) {
				if ((x + y) % 2 == 0) {
					return ((double) Math.max(xLeft, yLeft) + Math.min(xRight, yRight)) / 2;
				} else {
					return Math.max(xLeft, yLeft);
				}
			} else if (xLeft > yRight) {
				high = partX - 1;
			} else {
				low = partX + 1;
			}
		}
		return 0;
	}

	// Longest Palindromic Substring
	public String longestPalindrome(String s) {
		int start = 0;
		int end = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			int left = i;
			int right = i;

			while (left >= 0 && s.charAt(left) == c) {
				left--;
			}

			while (right < s.length() && s.charAt(right) == c) {
				right++;
			}

			while (left >= 0 && right < s.length()) {
				if (s.charAt(left) != s.charAt(right)) {
					break;
				}
				left--;
				right++;
			}

			left = left + 1;
			if (end - start < right - left) {
				start = left;
				end = right;
			}
		}

		return s.substring(start, end);
	}

	// Zigzag conversion of a string
	public String convert(String s, int numRows) {
		char[] c = s.toCharArray();
		int len = c.length;
		StringBuffer[] sb = new StringBuffer[numRows];
		for (int i = 0; i < sb.length; i++)
			sb[i] = new StringBuffer();

		int i = 0;
		while (i < len) {
			for (int idx = 0; idx < numRows && i < len; idx++)
				sb[idx].append(c[i++]);
			for (int idx = numRows - 2; idx >= 1 && i < len; idx--)
				sb[idx].append(c[i++]);
		}
		for (int idx = 1; idx < sb.length; idx++)
			sb[0].append(sb[idx]);
		return sb[0].toString();
	}

	// String to integere ATOI
	public int myAtoi(String str) {
		int index = 0, sign = 1, total = 0;
		if (str.length() == 0 || (str.trim().length() == 0)) {
			return index;
		}
		while (index < str.length() && str.charAt(index) == ' ')
			index++;
		if (str.charAt(index) == '+' || str.charAt(index) == '-') {
			sign = str.charAt(index) == '+' ? 1 : -1;
			index++;
		}
		while (index < str.length()) {
			int digit = str.charAt(index) - '0';
			if (digit < 0 || digit > 9)
				break;
			if (Integer.MAX_VALUE / 10 < total || Integer.MAX_VALUE / 10 == total && Integer.MAX_VALUE % 10 < digit)
				return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
			total = 10 * total + digit;
			index++;
		}
		return total * sign;
	}

	public List<List<Integer>> threeSum(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		Arrays.sort(nums);
		for (int i = 0; i + 2 < nums.length; i++) {
			if (i > 0 && nums[i] == nums[i - 1]) { // skip same result
				continue;
			}
			int j = i + 1, k = nums.length - 1;
			int target = -nums[i];
			while (j < k) {
				if (nums[j] + nums[k] == target) {
					res.add(Arrays.asList(nums[i], nums[j], nums[k]));
					j++;
					k--;
					while (j < k && nums[j] == nums[j - 1])
						j++; // skip same result
					while (j < k && nums[k] == nums[k + 1])
						k--; // skip same result
				} else if (nums[j] + nums[k] > target) {
					k--;
				} else {
					j++;
				}
			}
		}
		return res;
	}

	// Valid Parentheses for equation
	public boolean isValid(String s) {
		HashMap<Character, Character> brackets = new HashMap<>();
		brackets.put('(', ')');
		brackets.put('[', ']');
		brackets.put('{', '}');
		Deque<Character> stack = new LinkedList<>();

		for (Character c : s.toCharArray()) {
			if (brackets.containsKey(c))
				stack.push(c);
			else if (!c.equals(brackets.get(stack.poll())))
				return false;
		}
		return stack.isEmpty() ? true : false;
	}

	// Generate output string combinations based on input
	public List<String> generateParenthesis(int n) {
		List<String> list = new ArrayList<String>();
		generateOneByOne("", list, n, n);
		return list;
	}

	public void generateOneByOne(String sublist, List<String> list, int left, int right) {
		if (left > right) {
			return;
		}
		if (left > 0) {
			generateOneByOne(sublist + "(", list, left - 1, right);
		}
		if (right > 0) {
			generateOneByOne(sublist + ")", list, left, right - 1);
		}
		if (left == 0 && right == 0) {
			list.add(sublist);
			return;
		}
	}

	public ListNode mergeKLists(ListNode[] lists) {
		ListNode head = null, last = null;
		PriorityQueue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>() {
			public int compare(ListNode a, ListNode b) {
				return a.val - b.val;
			}
		});

		for (int i = 0; i < lists.length; i++)
			if (lists[i] != null)
				pq.add(lists[i]);
		while (!pq.isEmpty()) {
			ListNode top = pq.peek();
			pq.remove();
			if (top.next != null)
				pq.add(top.next);
			if (head == null) {
				head = top;
				last = top;
			} else {
				last.next = top;
				last = top;
			}
		}
		return head;
	}

	// Trap Rain Water
	public int trap(int[] height) {
		int n = height.length;
		int result = 0;
		int left_max = 0, right_max = 0;
		int lo = 0, hi = n - 1;

		while (lo <= hi) {
			if (height[lo] < height[hi]) {
				if (height[lo] > left_max)
					left_max = height[lo];
				else
					result += left_max - height[lo];
				lo++;
			} else {
				if (height[hi] > right_max)
					right_max = height[hi];
				else
					result += right_max - height[hi];
				hi--;
			}
		}
		return result;
	}

	// All possible permutations for int array
	public List<List<Integer>> permute(int[] nums) {
		List<List<Integer>> list = new ArrayList<>();
		ArrayList<Integer> perm = new ArrayList<Integer>();
		backTrack(perm, 0, nums, list);
		return list;
	}

	void backTrack(ArrayList<Integer> perm, int i, int[] nums, List<List<Integer>> list) {
		if (i == nums.length) {
			list.add(new ArrayList<Integer>(perm));
			return;
		}
		ArrayList<Integer> newPerm = new ArrayList<Integer>(perm);
		for (int j = 0; j <= i; j++) {
			newPerm.add(j, nums[i]);
			backTrack(newPerm, i + 1, nums, list);
			newPerm.remove(j);
		}
	}

	// Matrix rotation in place - anticlockwise
	public void rotateAnticlockwise(int[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = i; j < matrix[i].length; j++) {
				int temp = matrix[j][i];
				matrix[j][i] = matrix[i][j];
				matrix[i][j] = temp;
			}
		}

		for (int i = 0; i < matrix[0].length; i++) {
			for (int j = 0, k = matrix[i].length - 1; j < k; j++, k--) {
				int temp = matrix[j][i];
				matrix[j][i] = matrix[k][i];
				matrix[k][i] = temp;
			}
		}
	}

	// Rotate matrix in place - clockwise
	public void rotate(int[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = i; j < matrix[0].length; j++) {
				int temp = 0;
				temp = matrix[i][j];
				matrix[i][j] = matrix[j][i];
				matrix[j][i] = temp;
			}
		}
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix.length / 2; j++) {
				int temp = 0;
				temp = matrix[i][j];
				matrix[i][j] = matrix[i][matrix.length - 1 - j];
				matrix[i][matrix.length - 1 - j] = temp;
			}
		}
	}

	// Group anagrams of strings together
	public List<List<String>> groupAnagrams(String[] strs) {
		if (strs == null || strs.length == 0)
			return new ArrayList<>();
		Map<String, List<String>> map = new HashMap<>();
		for (String s : strs) {
			char[] ca = new char[26];
			for (char c : s.toCharArray())
				ca[c - 'a']++;
			String keyStr = String.valueOf(ca);
			if (!map.containsKey(keyStr))
				map.put(keyStr, new ArrayList<>());
			map.get(keyStr).add(s);
		}
		return new ArrayList<>(map.values());
	}

	// Merge overlapping intervals
	public int[][] merge(int[][] intervals) {
		if (intervals == null || intervals.length == 0)
			return intervals;
		Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
		LinkedList<int[]> mergedIntervals = new LinkedList<>();
		for (int[] curr : intervals) {
			if (mergedIntervals.isEmpty() || mergedIntervals.getLast()[1] < curr[0])
				mergedIntervals.add(curr);
			else
				mergedIntervals.getLast()[1] = Math.max(mergedIntervals.getLast()[1], curr[1]);
		}
		return mergedIntervals.toArray(new int[0][]);
	}

	public int[][] generateMatrix(int n) {
		int[][] result = new int[n][n];
		int r = n, c = n;
		int value = 0;
		int startRow = 0, startColumn = 0;
		while (startRow < r && startColumn < c) {
			for (int i = startColumn; i < c; i++) {
				value++;
				result[startRow][i] = value;
			}
			startRow++;
			for (int i = startRow; i < r; i++) {
				value++;
				result[i][c - 1] = value;
			}
			c--;
			if (startRow < r) {
				for (int i = c - 1; i >= startColumn; i--) {
					value++;
					result[r - 1][i] = value;
				}
				r--;
			}

			if (startColumn < c) {
				for (int i = r - 1; i >= startRow; i--) {
					value++;
					result[i][startColumn] = value;
				}
				startColumn++;
			}
		}
		return result;
	}

	// Robot problem - Unique paths to reach the bottom of a matrix
	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
		int m = obstacleGrid.length;
		int n = obstacleGrid[0].length;

		obstacleGrid[0][0] ^= 1;
		for (int i = 1; i < m; i++) {
			obstacleGrid[i][0] = (obstacleGrid[i][0] == 1) ? 0 : obstacleGrid[i - 1][0];
		}

		for (int j = 1; j < n; j++) {
			obstacleGrid[0][j] = (obstacleGrid[0][j] == 1) ? 0 : obstacleGrid[0][j - 1];
		}
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				obstacleGrid[i][j] = (obstacleGrid[i][j] == 1) ? 0 : obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
			}
		}
		return obstacleGrid[m - 1][n - 1];
	}

	// Minimum path sum in a matrix from top left to bottom right.
	public int minPathSum(int[][] grid) {
		int m = grid.length, n = grid[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (i == 0 && j != 0)
					grid[i][j] += grid[i][j - 1];
				if (i != 0 && j == 0)
					grid[i][j] += grid[i - 1][j];
				if (i != 0 && j != 0)
					grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
			}
		}
		return grid[m - 1][n - 1];
	}

	/*
	 * Given two strings word1 and word2, return the minimum number of operations
	 * required to convert word1 to word2.
	 * 
	 * You have the following three operations permitted on a word:
	 * 
	 * Insert a character Delete a character Replace a character
	 */
	public int minDistance(String word1, String word2) {
		if (word1.equals(word2)) {
			return 0;
		}
		if (word1.length() == 0 || word2.length() == 0) {
			return Math.abs(word1.length() - word2.length());
		}
		int[][] dp = new int[word1.length() + 1][word2.length() + 1];
		for (int i = 0; i <= word1.length(); i++) {
			dp[i][0] = i;
		}
		for (int i = 0; i <= word2.length(); i++) {
			dp[0][i] = i;
		}
		for (int i = 1; i <= word1.length(); i++) {
			for (int j = 1; j <= word2.length(); j++) {
				if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
					dp[i][j] = dp[i - 1][j - 1];
				} else {
					dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
				}
			}
		}
		return dp[word1.length()][word2.length()];
	}

	// Set entire row and column as zero if zero present
	public void setZeroes(int[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
			return;
		}
		int m = matrix.length;
		int n = matrix[0].length;
		boolean first_row = false;
		boolean first_col = false;
		for (int i = 0; i < m; i++) {
			if (matrix[i][0] == 0) {
				first_col = true;
				break;
			}
		}
		for (int j = 0; j < n; j++) {
			if (matrix[0][j] == 0) {
				first_row = true;
				break;
			}
		}
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				if (matrix[i][j] == 0) {
					matrix[i][0] = 0;
					matrix[0][j] = 0;
				}
			}
		}
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				if (matrix[i][0] == 0 || matrix[0][j] == 0) {
					matrix[i][j] = 0;
				}
			}
		}
		if (first_row) {
			for (int j = 0; j < n; j++) {
				matrix[0][j] = 0;
			}
		}
		if (first_col) {
			for (int i = 0; i < m; i++) {
				matrix[i][0] = 0;
			}
		}
	}

	// Sort and group colors together in O(1) space using 1-pass
	public void sortColors(int[] nums) {
		if (nums == null || nums.length <= 1) {
			return;
		}

		int zeros = 0;
		int twos = nums.length - 1;
		int i = 0;

		while (i <= twos) {
			if (nums[i] == 0) {
				swap(nums, i, zeros);
				zeros++;
				i++;
			} else if (nums[i] == 2) {
				swap(nums, i, twos);
				twos--;
			} else {
				i++;
			}
		}
	}

	private void swap(int[] nums, int i, int j) {
		if (i != j) {
			int temp = nums[i];
			nums[i] = nums[j];
			nums[j] = temp;
		}
	}

	/*
	 * Minimum Window Substring Given two strings s and t of lengths m and n
	 * respectively, return the minimum window substring of s such that every
	 * character in t (including duplicates) is included in the window. If there is
	 * no such substring, return the empty string "".
	 * 
	 * The testcases will be generated such that the answer is unique.
	 * 
	 * A substring is a contiguous sequence of characters within the string.
	 */

	public String minWindow(String s, String t) {
		if (s == null || t == null || s.length() < t.length() || t.length() == 0) {
			return "";
		}

		HashMap<Character, Integer> map = new HashMap<>();
		for (int i = 0; i < t.length(); i++) {
			map.put(t.charAt(i), map.getOrDefault(t.charAt(i), 0) + 1);
		}

		int start = 0;
		int end = 0;
		int charTLeft = t.length();
		int minStart = 0;
		int minLen = Integer.MAX_VALUE;

		while (end < s.length()) {
			char eChar = s.charAt(end);
			if (map.containsKey(eChar)) {
				int count = map.get(eChar);
				if (count > 0) {
					charTLeft--;
				}
				map.put(eChar, count - 1);
			}
			end++;

			while (charTLeft == 0) {
				if (minLen > end - start) {
					minLen = end - start;
					minStart = start;
				}
				char sChar = s.charAt(start);
				if (map.containsKey(sChar)) {
					int count = map.get(sChar);
					if (count == 0) {
						charTLeft++;
					}
					map.put(sChar, count + 1);
				}
				start++;
			}
		}

		return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
	}

	// Subsets
	/*
	 * Given an integer array nums of unique elements, return all possible subsets
	 * (the power set).
	 * 
	 * The solution set must not contain duplicate subsets. Return the solution in
	 * any order.
	 */
	public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> list = new ArrayList<>();
		Arrays.sort(nums);
		backtrack(list, new ArrayList<>(), nums, 0);
		return list;
	}

	private void backtrack(List<List<Integer>> list, List<Integer> tempList, int[] nums, int start) {
		list.add(new ArrayList<>(tempList));
		for (int i = start; i < nums.length; i++) {
			tempList.add(nums[i]);
			backtrack(list, tempList, nums, i + 1);
			tempList.remove(tempList.size() - 1);
		}
	}

	// Gray code
	/*
	 * An n-bit gray code sequence is a sequence of 2n integers where:
	 * 
	 * Every integer is in the inclusive range [0, 2n - 1], The first integer is 0,
	 * An integer appears no more than once in the sequence, The binary
	 * representation of every pair of adjacent integers differs by exactly one bit,
	 * and The binary representation of the first and last integers differs by
	 * exactly one bit.
	 */
	public List<Integer> grayCode(int n) {
		List<Integer> result = new LinkedList<>();
		for (int i = 0; i < 1 << n; i++)
			result.add(i ^ i >> 1);
		return result;
	}

	// Validate if left is lower than node, right is highe than node
	public boolean isValidBST(TreeNode root) {
		Stack<TreeNode> stack = new Stack<TreeNode>();
		TreeNode cur = root;
		TreeNode pre = null;
		while (!stack.isEmpty() || cur != null) {
			if (cur != null) {
				stack.push(cur);
				cur = cur.left;
			} else {
				TreeNode p = stack.pop();
				if (pre != null && p.val <= pre.val) {
					return false;
				}
				pre = p;
				cur = p.right;
			}
		}
		return true;
	}

	// Validate if two Binary trees are equal
	public boolean isSameTree(TreeNode p, TreeNode q) {
		if ((p == null && q == null)) {
			return true;
		} else if (p != null && q != null) {
			return ((p.val == q.val) && isSameTree(p.left, q.left) && isSameTree(p.right, q.right));
		}
		return false;
	}

	// Level Order traversal of a BT
	public List<List<Integer>> levelOrder(TreeNode root) {
		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		List<List<Integer>> wrapList = new LinkedList<List<Integer>>();

		if (root == null)
			return wrapList;

		queue.offer(root);
		while (!queue.isEmpty()) {
			int levelNum = queue.size();
			List<Integer> subList = new LinkedList<Integer>();
			for (int i = 0; i < levelNum; i++) {
				if (queue.peek().left != null)
					queue.offer(queue.peek().left);
				if (queue.peek().right != null)
					queue.offer(queue.peek().right);
				subList.add(queue.poll().val);
			}
			wrapList.add(subList);
		}
		return wrapList;
	}

	// Path sum from root to leaf equal to a given no - using BFS
	public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
		List<List<Integer>> ans = new ArrayList<>();
		List<Integer> path = new ArrayList<>();
		pathSum(ans, path, root, targetSum);
		return ans;
	}

	private void pathSum(List<List<Integer>> ans, List<Integer> path, TreeNode root, int sum) {
		if (root == null)
			return;
		List<Integer> newPath = new ArrayList<>(path);
		newPath.add(root.val);
		if (root.left == null && root.right == null && root.val == sum) {
			ans.add(newPath);
			return;
		}
		pathSum(ans, newPath, root.left, sum - root.val);
		pathSum(ans, newPath, root.right, sum - root.val);
	}

	// Pascal's Triangle II
	/*
	 * Given an integer rowIndex, return the rowIndexth (0-indexed) row of the
	 * Pascal's triangle.
	 * 
	 * In Pascal's triangle, each number is the sum of the two numbers directly
	 * above it as shown:
	 */
	public List<Integer> getRow(int rowIndex) {
		List<Integer> ret = new LinkedList<Integer>();
		if (rowIndex < 0) {
			return ret;
		}
		for (int row = 0; row <= rowIndex; row++) {
			ret.add(0, 1);
			for (int i = 1; i < row; i++)
				ret.set(i, ret.get(i) + ret.get(i + 1));
		}
		return ret;
	}

	// Best Time to Buy and Sell Stock
	/*
	 * You are given an array prices where prices[i] is the price of a given stock
	 * on the ith day.
	 * 
	 * You want to maximize your profit by choosing a single day to buy one stock
	 * and choosing a different day in the future to sell that stock.
	 * 
	 * Return the maximum profit you can achieve from this transaction. If you
	 * cannot achieve any profit, return 0.
	 */
	// Kadane's Algo
	public int maxProfit(int[] prices) {
		int maxCur = 0, maxSoFar = 0;
		for (int i = 1; i < prices.length; i++) {
			maxCur = Math.max(0, maxCur += prices[i] - prices[i - 1]);
			maxSoFar = Math.max(maxCur, maxSoFar);
		}
		return maxSoFar;
	}

	// Sum Root to Leaf Numbers
	/*
	 * You are given the root of a binary tree containing digits from 0 to 9 only.
	 * 
	 * Each root-to-leaf path in the tree represents a number.
	 * 
	 * For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.
	 * Return the total sum of all root-to-leaf numbers. Test cases are generated so
	 * that the answer will fit in a 32-bit integer.
	 * 
	 * A leaf node is a node with no children.
	 */
	public int sumNumbers(TreeNode root) {
		return sum(root, 0);
	}

	public int sum(TreeNode n, int s) {
		if (n == null)
			return 0;
		if (n.right == null && n.left == null)
			return s * 10 + n.val;
		return sum(n.left, s * 10 + n.val) + sum(n.right, s * 10 + n.val);
	}

	// Copy List with Random Pointer
	/*
	 * A linked list of length n is given such that each node contains an additional
	 * random pointer, which could point to any node in the list, or null.
	 * 
	 * Construct a deep copy of the list. The deep copy should consist of exactly n
	 * brand new nodes, where each new node has its value set to the value of its
	 * corresponding original node. Both the next and random pointer of the new
	 * nodes should point to new nodes in the copied list such that the pointers in
	 * the original list and copied list represent the same list state. None of the
	 * pointers in the new list should point to nodes in the original list.
	 */
	public Node copyRandomList(Node head) {
		HashMap<Node, Node> map = new HashMap<Node, Node>();
		Node p = head;
		while (p != null) {
			map.put(p, new Node(p.val));
			p = p.next;
		}
		Node q = head;
		while (q != null) {
			map.get(q).next = map.get(q.next);
			map.get(q).random = map.get(q.random);
			q = q.next;
		}
		return map.get(head);
	}

	// Word Break
	/*
	 * Given a string s and a dictionary of strings wordDict, return true if s can
	 * be segmented into a space-separated sequence of one or more dictionary words.
	 * 
	 * Note that the same word in the dictionary may be reused multiple times in the
	 * segmentation.
	 */
	public boolean wordBreak(String s, List<String> wordDict) {
		TrieNode t = new TrieNode(), cur;
		for (String i : wordDict)
			addWord(t, i);
		char[] str = s.toCharArray();
		int len = str.length;
		boolean[] f = new boolean[len + 1];
		f[len] = true;

		for (int i = len - 1; i >= 0; i--) {
			// System.out.println(str[i]);
			cur = t;
			for (int j = i; cur != null && j < len; j++) {
				cur = cur.c[(int) str[j]];
				if (cur != null && cur.isWord && f[j + 1]) {
					f[i] = true;
					break;
				}
			}
		}
		return f[0];
	}

	public void addWord(TrieNode t, String w) {
		for (int i = 0; i < w.length(); i++) {
			int j = (int) w.charAt(i);
			if (t.c[j] == null)
				t.c[j] = new TrieNode();
			t = t.c[j];
		}
		t.isWord = true;
	}

	// Linked List Cycle
	/*
	 * Given head, the head of a linked list, determine if the linked list has a
	 * cycle in it.
	 * 
	 * There is a cycle in a linked list if there is some node in the list that can
	 * be reached again by continuously following the next pointer. Internally, pos
	 * is used to denote the index of the node that tail's next pointer is connected
	 * to. Note that pos is not passed as a parameter.
	 * 
	 * Return true if there is a cycle in the linked list. Otherwise, return false.
	 */
	public boolean hasCycle(ListNode head) {
		ListNode slow_p = head, fast_p = head;
		int flag = 0;
		while (slow_p != null && fast_p != null && fast_p.next != null) {
			slow_p = slow_p.next;
			fast_p = fast_p.next.next;
			if (slow_p == fast_p) {
				flag = 1;
				break;
			}
		}
		return flag == 1 ? true : false;
	}

	// Sort a Linked List
	/*
	 * Given the head of a linked list, return the list after sorting it in
	 * ascending order.
	 */
	public ListNode sortList(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode mid = findMid(head);
		ListNode head2 = mid.next;
		mid.next = null;
		ListNode newHead1 = sortList(head);
		ListNode newHead2 = sortList(head2);
		ListNode finalHead = merge(newHead1, newHead2);
		return finalHead;
	}

	static ListNode merge(ListNode head1, ListNode head2) {
		ListNode merged = new ListNode(-1);
		ListNode temp = merged;
		while (head1 != null && head2 != null) {
			if (head1.val < head2.val) {
				temp.next = head1;
				head1 = head1.next;
			} else {
				temp.next = head2;
				head2 = head2.next;
			}
			temp = temp.next;
		}
		while (head1 != null) {
			temp.next = head1;
			head1 = head1.next;
			temp = temp.next;
		}
		while (head2 != null) {
			temp.next = head2;
			head2 = head2.next;
			temp = temp.next;
		}
		return merged.next;
	}

	private static ListNode findMid(ListNode head) {
		ListNode slow = head, fast = head.next;
		while (fast != null && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		return slow;
	}

	// Intersection of two linked lists
	public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
		ListNode ptr1 = headA;
		ListNode ptr2 = headB;
		if (ptr1 == null || ptr2 == null) {
			return null;
		}
		while (ptr1 != ptr2) {
			ptr1 = ptr1.next;
			ptr2 = ptr2.next;
			if (ptr1 == ptr2) {
				return ptr1;
			}
			if (ptr1 == null) {
				ptr1 = headB;
			}
			if (ptr2 == null) {
				ptr2 = headA;
			}
		}
		return ptr1;
	}

	// Two Sum II - Input Array Is Sorted
	/*
	 * Given a 1-indexed array of integers numbers that is already sorted in
	 * non-decreasing order, find two numbers such that they add up to a specific
	 * target number. Let these two numbers be numbers[index1] and numbers[index2]
	 * where 1 <= index1 < index2 <= numbers.length.
	 * 
	 * Return the indices of the two numbers, index1 and index2, added by one as an
	 * integer array [index1, index2] of length 2.
	 * 
	 * The tests are generated such that there is exactly one solution. You may not
	 * use the same element twice.
	 */
	public int[] twoSumTwo(int[] numbers, int target) {
		int[] result = new int[2];
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < numbers.length; i++) {
			if (map.containsKey(target - numbers[i])) {
				result[1] = i + 1;
				result[0] = map.get(target - numbers[i]) + 1;
				return result;
			}
			map.put(numbers[i], i);
		}
		return result;
	}

	// Rotate array to the right
	public void rotate(int[] nums, int k) {
		if (nums == null || nums.length < 2) {
			return;
		}

		k = k % nums.length;
		reverse(nums, 0, nums.length - k - 1);
		reverse(nums, nums.length - k, nums.length - 1);
		reverse(nums, 0, nums.length - 1);
	}

	private void reverse(int[] nums, int i, int j) {
		int tmp = 0;
		while (i < j) {
			tmp = nums[i];
			nums[i] = nums[j];
			nums[j] = tmp;
			i++;
			j--;
		}
	}

	// Number of Islands
	/*
	 * Given an m x n 2D binary grid grid which represents a map of '1's (land) and
	 * '0's (water), return the number of islands.
	 * 
	 * An island is surrounded by water and is formed by connecting adjacent lands
	 * horizontally or vertically. You may assume all four edges of the grid are all
	 * surrounded by water.
	 * 
	 * 
	 * 
	 * Example 1:
	 * 
	 * Input: grid = [ ["1","1","1","1","0"], ["1","1","0","1","0"],
	 * ["1","1","0","0","0"], ["0","0","0","0","0"] ] Output: 1
	 */
	public int numIslands(char[][] grid) {
		int count = 0;
		for (int i = 0; i < grid.length; i++) {
			for (int j = 0; j < grid[0].length; j++) {
				if (grid[i][j] == '1') {
					dfsFill(grid, i, j);
					count++;
				}
			}
		}
		return count;
	}

	private void dfsFill(char[][] grid, int i, int j) {
		if (i >= 0 && j >= 0 && i < grid.length && j < grid[0].length && grid[i][j] == '1') {
			grid[i][j] = '0';
			dfsFill(grid, i + 1, j);
			dfsFill(grid, i - 1, j);
			dfsFill(grid, i, j + 1);
			dfsFill(grid, i, j - 1);
		}
	}

	// Happy Number
	/*
	 * Write an algorithm to determine if a number n is happy.
	 * 
	 * A happy number is a number defined by the following process:
	 * 
	 * Starting with any positive integer, replace the number by the sum of the
	 * squares of its digits. Repeat the process until the number equals 1 (where it
	 * will stay), or it loops endlessly in a cycle which does not include 1. Those
	 * numbers for which this process ends in 1 are happy. Return true if n is a
	 * happy number, and false if not.
	 */
	public boolean isHappy(int n) {
		Set<Integer> inLoop = new HashSet<Integer>();
		int squareSum, remain;
		while (inLoop.add(n)) {
			squareSum = 0;
			while (n > 0) {
				remain = n % 10;
				squareSum += remain * remain;
				n /= 10;
			}
			if (squareSum == 1)
				return true;
			else
				n = squareSum;

		}
		return false;
	}

	// Count primes less than N
	/*
	 * Given an integer n, return the number of prime numbers that are strictly less
	 * than n.
	 */
	public int countPrimes(int n) {
		if (n < 3)
			return 0;

		boolean[] f = new boolean[n];
		int count = n / 2;
		for (int i = 3; i * i < n; i += 2) {
			if (f[i])
				continue;

			for (int j = i * i; j < n; j += 2 * i) {
				if (!f[j]) {
					--count;
					f[j] = true;
				}
			}
		}
		return count;
	}

	// Reverse a linked list
	public ListNode reverseList(ListNode head) {
		ListNode prev = null;
		ListNode current = head;
		ListNode next = null;
		while (current != null) {
			next = current.next;
			current.next = prev;
			prev = current;
			current = next;
		}
		head = prev;
		return head;
	}

	// Word Search
	/*
	 * Given an m x n board of characters and a list of strings words, return all
	 * words on the board.
	 * 
	 * Each word must be constructed from letters of sequentially adjacent cells,
	 * where adjacent cells are horizontally or vertically neighboring. The same
	 * letter cell may not be used more than once in a word. Input: board =
	 * [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],
	 * words = ["oath","pea","eat","rain"] Output: ["eat","oath"]
	 */
	public List<String> findWords(char[][] board, String[] words) {
		List<String> res = new ArrayList<>();
		TrieNode root = buildTrie(words);
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[0].length; j++) {
				dfs(board, i, j, root, res);
			}
		}
		return res;
	}

	public void dfs(char[][] board, int i, int j, TrieNode p, List<String> res) {
		char c = board[i][j];
		if (c == '#' || p.next[c - 'a'] == null)
			return;
		p = p.next[c - 'a'];
		if (p.word != null) { // found one
			res.add(p.word);
			p.word = null; // de-duplicate
		}

		board[i][j] = '#';
		if (i > 0)
			dfs(board, i - 1, j, p, res);
		if (j > 0)
			dfs(board, i, j - 1, p, res);
		if (i < board.length - 1)
			dfs(board, i + 1, j, p, res);
		if (j < board[0].length - 1)
			dfs(board, i, j + 1, p, res);
		board[i][j] = c;
	}

	public TrieNode buildTrie(String[] words) {
		TrieNode root = new TrieNode();
		for (String w : words) {
			TrieNode p = root;
			for (char c : w.toCharArray()) {
				int i = c - 'a';
				if (p.next[i] == null)
					p.next[i] = new TrieNode();
				p = p.next[i];
			}
			p.word = w;
		}
		return root;
	}

	// Kth Largest Element in an Array
	/*
	 * Given an integer array nums and an integer k, return the kth largest element
	 * in the array.
	 * 
	 * Note that it is the kth largest element in the sorted order, not the kth
	 * distinct element.
	 */
	public int findKthLargest(int[] nums, int k) {
		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(k + 1);
		for (int el : nums) {
			pq.add(el);
			if (pq.size() > k) {
				pq.poll();
			}
		}
		return pq.poll();
	}

	// Basic calculator
	/*
	 * Given a string s representing a valid expression, implement a basic
	 * calculator to evaluate it, and return the result of the evaluation.
	 * 
	 * Note: You are not allowed to use any built-in function which evaluates
	 * strings as mathematical expressions, such as eval().
	 */
	public int calculate(String s) {
		int len = s.length(), sign = 1, result = 0;
		Stack<Integer> stack = new Stack<Integer>();
		for (int i = 0; i < len; i++) {
			if (Character.isDigit(s.charAt(i))) {
				int sum = s.charAt(i) - '0';
				while (i + 1 < len && Character.isDigit(s.charAt(i + 1))) {
					sum = sum * 10 + s.charAt(i + 1) - '0';
					i++;
				}
				result += sum * sign;
			} else if (s.charAt(i) == '+')
				sign = 1;
			else if (s.charAt(i) == '-')
				sign = -1;
			else if (s.charAt(i) == '(') {
				stack.push(result);
				stack.push(sign);
				result = 0;
				sign = 1;
			} else if (s.charAt(i) == ')') {
				result = result * stack.pop() + stack.pop();
			}

		}
		return result;
	}

	// Palindrome Linked List
	// Given the head of a singly linked list, return true if it is a palindrome.
	public boolean isPalindrome(ListNode head) {
		ListNode slow_ptr = head;
		ListNode fast_ptr = head;
		ListNode prev_of_slow_ptr = head;
		ListNode midnode = null;
		ListNode second_half = null;
		boolean res = true;

		if (head != null && head.next != null) {
			while (fast_ptr != null && fast_ptr.next != null) {
				fast_ptr = fast_ptr.next.next;
				prev_of_slow_ptr = slow_ptr;
				slow_ptr = slow_ptr.next;
			}
			if (fast_ptr != null) {
				midnode = slow_ptr;
				slow_ptr = slow_ptr.next;
			}
			second_half = slow_ptr;
			prev_of_slow_ptr.next = null;
			second_half = reverseList(second_half);
			res = compareLists(head, second_half);

			second_half = reverseList(second_half);

			if (midnode != null) {
				prev_of_slow_ptr.next = midnode;
				midnode.next = second_half;
			} else
				prev_of_slow_ptr.next = second_half;
		}
		return res;
	}

	private static boolean compareLists(ListNode head1, ListNode head2) {
		ListNode temp1 = head1;
		ListNode temp2 = head2;
		while (temp1 != null && temp2 != null) {
			if (temp1.val == temp2.val) {
				temp1 = temp1.next;
				temp2 = temp2.next;
			} else
				return false;
		}
		if (temp1 == null && temp2 == null)
			return true;
		return false;
	}

	// Lowest Common Ancestor of a Binary Search Tree/Binary Tree
	/*
	 * Given a binary search tree (BST), find the lowest common ancestor (LCA) of
	 * two given nodes in the BST.
	 * 
	 * According to the definition of LCA on Wikipedia: “The lowest common ancestor
	 * is defined between two nodes p and q as the lowest node in T that has both p
	 * and q as descendants (where we allow a node to be a descendant of itself).”
	 */
	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		if (root == null) {
			return null;
		}

		if (root.val == p.val || root.val == q.val) {
			return root;
		}

		TreeNode leftLCA = lowestCommonAncestor(root.left, p, q);
		TreeNode rightLCA = lowestCommonAncestor(root.right, p, q);

		if (leftLCA != null && rightLCA != null) {
			return root;
		}

		return leftLCA != null ? leftLCA : rightLCA;
	}

	// Product of Array Except Self
	/*
	 * Given an integer array nums, return an array answer such that answer[i] is
	 * equal to the product of all the elements of nums except nums[i].
	 * 
	 * The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit
	 * integer.
	 * 
	 * You must write an algorithm that runs in O(n) time and without using the
	 * division operation.
	 */
	public int[] productExceptSelf(int[] nums) {
		int n = nums.length;
		int[] res = new int[n];
		res[0] = 1;
		for (int i = 1; i < n; i++) {
			res[i] = res[i - 1] * nums[i - 1];
		}
		int right = 1;
		for (int i = n - 1; i >= 0; i--) {
			res[i] *= right;
			right *= nums[i];
		}
		return res;
	}

	// Sliding Window Maximum
	/*
	 * You are given an array of integers nums, there is a sliding window of size k
	 * which is moving from the very left of the array to the very right. You can
	 * only see the k numbers in the window. Each time the sliding window moves
	 * right by one position.
	 * 
	 * Return the max sliding window.
	 */
	public int[] maxSlidingWindow(int[] nums, int k) {
		if (nums == null || k <= 0 || k > nums.length) {
			return new int[0];
		}
		int[] result = new int[nums.length - k + 1];
		int ri = 0;
		Deque<Integer> q = new ArrayDeque<>();
		for (int i = 0; i < nums.length; i++) {
			while (!q.isEmpty() && q.peek() < i - k + 1) {
				q.poll();
			}
			while (!q.isEmpty() && nums[q.peekLast()] < nums[i]) {
				q.pollLast();
			}
			q.offer(i);
			if (i >= k - 1) {
				result[ri++] = nums[q.peek()];
			}
		}
		return result;
	}

	// Search a 2D Matrix II
	/*
	 * Write an efficient algorithm that searches for a target value in an m x n
	 * integer matrix. The matrix has the following properties:
	 * 
	 * Integers in each row are sorted in ascending from left to right. Integers in
	 * each column are sorted in ascending from top to bottom.
	 */
	public boolean searchMatrix(int[][] matrix, int target) {
		if (matrix == null) {
			return false;
		}
		if (matrix.length == 0 || matrix[0].length == 0) {
			return false;
		}

		int row = 0;
		int col = matrix[0].length - 1;

		while (row < matrix.length && col >= 0) {
			if (matrix[row][col] == target) {
				return true;
			}
			if (target < matrix[row][col]) {
				col--;
			} else {
				row++;
			}
		}

		return false;
	}

	// Valid Anagram
	/*
	 * Given two strings s and t, return true if t is an anagram of s, and false
	 * otherwise.
	 * 
	 * An Anagram is a word or phrase formed by rearranging the letters of a
	 * different word or phrase, typically using all the original letters exactly
	 * once.
	 */
	public boolean isAnagram(String s, String t) {
		if (s == null || t == null || s.length() != t.length())
			return false;
		int[] count = new int[26];
		int len = t.length();
		for (int i = 0; i < len; i++) {
			count[t.charAt(i) - 'a']++;
		}
		for (int i = 0; i < len; i++) {
			char c = s.charAt(i);
			if (count[c - 'a'] > 0) {
				count[c - 'a']--;
			} else {
				return false;
			}
		}
		return true;
	}

	// Top K Frequent Elements
	/*
	 * Given an integer array nums and an integer k, return the k most frequent
	 * elements. You may return the answer in any order.
	 */
	public int[] topKFrequent(int[] nums, int k) {
		Map<Integer, Integer> mp = new HashMap<Integer, Integer>();
		int[] result = new int[k];

		for (int i = 0; i < nums.length; i++) {
			mp.put(nums[i], mp.getOrDefault(nums[i], 0) + 1);
		}

		PriorityQueue<Map.Entry<Integer, Integer>> queue = new PriorityQueue<>(
				(a, b) -> a.getValue().equals(b.getValue()) ? Integer.compare(b.getKey(), a.getKey())
						: Integer.compare(b.getValue(), a.getValue()));

		for (Map.Entry<Integer, Integer> entry : mp.entrySet()) {
			queue.offer(entry);
		}

		for (int i = 0; i < k; i++) {
			result[i] = queue.poll().getKey();
		}

		return result;
	}

	// First Unique Character in a String
	/*
	 * Given a string s, find the first non-repeating character in it and return its
	 * index. If it does not exist, return -1.
	 */
	public int firstUniqChar(String s) {
		int freq[] = new int[26];
		for (int i = 0; i < s.length(); i++)
			freq[s.charAt(i) - 'a']++;
		for (int i = 0; i < s.length(); i++)
			if (freq[s.charAt(i) - 'a'] == 1)
				return i;
		return -1;
	}

	// Rotate Function
	/*
	 * You are given an integer array nums of length n.
	 * 
	 * Assume arrk to be an array obtained by rotating nums by k positions
	 * clock-wise. We define the rotation function F on nums as follow:
	 * 
	 * F(k) = 0 * arrk[0] + 1 * arrk[1] + ... + (n - 1) * arrk[n - 1]. Return the
	 * maximum value of F(0), F(1), ..., F(n-1).
	 */
	public int maxRotateFunction(int[] nums) {
		int n = nums.length;
		int sum = 0;
		int first = 0;

		for (int i = 0; i < n; i++) {
			sum += nums[i];
			first += nums[i] * i;
		}

		int max = first;

		for (int i = n - 1; i > 0; i--) {
			first = first + sum - nums[i] * n;
			max = Math.max(max, first);
		}
		return max;
	}

	// Third Maximum Number
	/*
	 * Given an integer array nums, return the third distinct maximum number in this
	 * array. If the third maximum does not exist, return the maximum number.
	 */
	public int thirdMax(int[] nums) {
		PriorityQueue<Integer> pq = new PriorityQueue<>();
		Set<Integer> set = new HashSet<>();
		for (int i : nums) {
			if (!set.contains(i)) {
				pq.offer(i);
				set.add(i);
				if (pq.size() > 3) {
					set.remove(pq.poll());
				}
			}
		}
		if (pq.size() < 3) {
			while (pq.size() > 1) {
				pq.poll();
			}
		}
		return pq.peek();
	}

	// Battleships in a Board
	/*
	 * Given an m x n matrix board where each cell is a battleship 'X' or empty '.',
	 * return the number of the battleships on board.
	 * 
	 * Battleships can only be placed horizontally or vertically on board. In other
	 * words, they can only be made of the shape 1 x k (1 row, k columns) or k x 1
	 * (k rows, 1 column), where k can be of any size. At least one horizontal or
	 * vertical cell separates between two battleships (i.e., there are no adjacent
	 * battleships).
	 */
	public int countBattleships(char[][] board) {
		if (board == null || board.length == 0 || board[0].length == 0)
			return 0;
		int R = board.length, C = board[0].length, cnt = 0;
		for (int i = 0; i < R; i++) {
			for (int j = 0; j < C; j++) {
				if (board[i][j] == 'X' && (i == 0 || board[i - 1][j] == '.') && (j == 0 || board[i][j - 1] == '.'))
					cnt++;
			}
		}
		return cnt;
	}

	// Find starting index of all Anagrams in a String
	/*
	 * Given two strings s and p, return an array of all the start indices of p's
	 * anagrams in s. You may return the answer in any order.
	 * 
	 * An Anagram is a word or phrase formed by rearranging the letters of a
	 * different word or phrase, typically using all the original letters exactly
	 * once.
	 */
	public List<Integer> findAnagrams(String s, String p) {
		ArrayList<Integer> soln = new ArrayList<Integer>();

		if (s.length() == 0 || p.length() == 0 || s.length() < p.length()) {
			return soln;
		}

		int[] chars = new int[26];
		for (Character c : p.toCharArray()) {
			chars[c - 'a']++;
		}

		int start = 0, end = 0, len = p.length(), diff = len;

		char temp;
		for (end = 0; end < len; end++) {
			temp = s.charAt(end);
			chars[temp - 'a']--;
			if (chars[temp - 'a'] >= 0) {
				diff--;
			}
		}

		if (diff == 0) {
			soln.add(0);
		}

		while (end < s.length()) {
			temp = s.charAt(start);

			if (chars[temp - 'a'] >= 0) {
				diff++;
			}

			chars[temp - 'a']++;
			start++;
			temp = s.charAt(end);
			chars[temp - 'a']--;

			if (chars[temp - 'a'] >= 0) {
				diff--;
			}
			if (diff == 0) {
				soln.add(start);
			}
			end++;
		}
		return soln;
	}

	// String Compression
	/*
	 * Given an array of characters chars, compress it using the following
	 * algorithm:
	 * 
	 * Begin with an empty string s. For each group of consecutive repeating
	 * characters in chars:
	 * 
	 * If the group's length is 1, append the character to s. Otherwise, append the
	 * character followed by the group's length. The compressed string s should not
	 * be returned separately, but instead, be stored in the input character array
	 * chars. Note that group lengths that are 10 or longer will be split into
	 * multiple characters in chars.
	 * 
	 * After you are done modifying the input array, return the new length of the
	 * array.
	 * 
	 * You must write an algorithm that uses only constant extra space.
	 */
	public int compress(char[] chars) {
		int indexAns = 0, index = 0;
		while (index < chars.length) {
			char currentChar = chars[index];
			int count = 0;
			while (index < chars.length && chars[index] == currentChar) {
				index++;
				count++;
			}
			chars[indexAns++] = currentChar;
			if (count != 1)
				for (char c : Integer.toString(count).toCharArray())
					chars[indexAns++] = c;
		}
		return indexAns;
	}

	// Sort Characters By Frequency
	/*
	 * Given a string s, sort it in decreasing order based on the frequency of the
	 * characters. The frequency of a character is the number of times it appears in
	 * the string.
	 * 
	 * Return the sorted string. If there are multiple answers, return any of them.
	 */
	// Bucket sort
	public String frequencySort(String s) {
		if (s.length() < 3)
			return s;
		int max = 0;
		int[] map = new int[256];
		for (char ch : s.toCharArray()) {
			map[ch]++;
			max = Math.max(max, map[ch]);
		}
		String[] buckets = new String[max + 1];
		for (int i = 0; i < 256; i++) {
			String str = buckets[map[i]];
			if (map[i] > 0)
				buckets[map[i]] = (str == null) ? "" + (char) i : (str + (char) i);
		}
		StringBuilder strb = new StringBuilder();
		for (int i = max; i >= 0; i--) {
			if (buckets[i] != null)
				for (char ch : buckets[i].toCharArray())
					for (int j = 0; j < i; j++)
						strb.append(ch);
		}
		return strb.toString();
	}

	// Repeated Substring Pattern
	/*
	 * Given a string s, check if it can be constructed by taking a substring of it
	 * and appending multiple copies of the substring together.
	 */
	public boolean repeatedSubstringPattern(String s) {
		int i = 1;
		while (i <= s.length() / 2) {
			if (s.charAt(0) == s.charAt(i) && s.length() % i == 0) {
				String s1 = s.substring(0, i);
				s1 = s1 + s1.repeat((s.length() / i) - 1);
				if (s.equals(s1)) {
					return true;
				}
			}
			i += 1;
		}
		return false;
	}

	// Concatenated Words
	/*
	 * Given an array of strings words (without duplicates), return all the
	 * concatenated words in the given list of words.
	 * 
	 * A concatenated word is defined as a string that is comprised entirely of at
	 * least two shorter words in the given array.
	 */
	public List<String> findAllConcatenatedWordsInADict(String[] words) {
		List<String> result = new ArrayList<>();
		Set<String> preWords = new HashSet<>();
		Arrays.sort(words, new Comparator<String>() {
			public int compare(String s1, String s2) {
				return s1.length() - s2.length();
			}
		});

		for (int i = 0; i < words.length; i++) {
			if (canForm(words[i], preWords)) {
				result.add(words[i]);
			}
			preWords.add(words[i]);
		}
		return result;
	}

	private static boolean canForm(String word, Set<String> dict) {
		if (dict.isEmpty())
			return false;
		boolean[] dp = new boolean[word.length() + 1];
		dp[0] = true;
		for (int i = 1; i <= word.length(); i++) {
			for (int j = 0; j < i; j++) {
				if (!dp[j])
					continue;
				if (dict.contains(word.substring(j, i))) {
					dp[i] = true;
					break;
				}
			}
		}
		return dp[word.length()];
	}

	// Longest Palindromic Subsequence
	/*
	 * Given a string s, find the longest palindromic subsequence's length in s.
	 * 
	 * A subsequence is a sequence that can be derived from another sequence by
	 * deleting some or no elements without changing the order of the remaining
	 * elements.
	 */
	public int longestPalindromeSubseq(String s) {
		int[][] dp = new int[s.length()][s.length()];

		for (int i = s.length() - 1; i >= 0; i--) {
			dp[i][i] = 1;
			for (int j = i + 1; j < s.length(); j++) {
				if (s.charAt(i) == s.charAt(j)) {
					dp[i][j] = dp[i + 1][j - 1] + 2;
				} else {
					dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
				}
			}
		}

		return dp[0][s.length() - 1];
	}

	// Super Washing Machines
	/*
	 * You have n super washing machines on a line. Initially, each washing machine
	 * has some dresses or is empty.
	 * 
	 * For each move, you could choose any m (1 <= m <= n) washing machines, and
	 * pass one dress of each washing machine to one of its adjacent washing
	 * machines at the same time.
	 * 
	 * Given an integer array machines representing the number of dresses in each
	 * washing machine from left to right on the line, return the minimum number of
	 * moves to make all the washing machines have the same number of dresses. If it
	 * is not possible to do it, return -1.
	 */
	public int findMinMoves(int[] machines) {
		int total = 0;
		int n = machines.length;
		for (int i = 0; i < n; i++) {
			total += machines[i];
		}
		if (total % n != 0) {
			return -1;
		}
		int result = 0, cur = 0;
		int finalCount = total / n;
		for (int i = 0; i < n; i++) {
			int diff = machines[i] - finalCount;
			result = Math.max(result, diff);
			cur += diff;
			result = Math.max(result, Math.abs(cur));
		}
		return result;
	}

	// Minesweeper
	/*
	 * Let's play the minesweeper game (Wikipedia, online game)!
	 * 
	 * You are given an m x n char matrix board representing the game board where:
	 * 
	 * 'M' represents an unrevealed mine, 'E' represents an unrevealed empty square,
	 * 'B' represents a revealed blank square that has no adjacent mines (i.e.,
	 * above, below, left, right, and all 4 diagonals), digit ('1' to '8')
	 * represents how many mines are adjacent to this revealed square, and 'X'
	 * represents a revealed mine. You are also given an integer array click where
	 * click = [clickr, clickc] represents the next click position among all the
	 * unrevealed squares ('M' or 'E').
	 * 
	 * Return the board after revealing this position according to the following
	 * rules:
	 * 
	 * If a mine 'M' is revealed, then the game is over. You should change it to
	 * 'X'. If an empty square 'E' with no adjacent mines is revealed, then change
	 * it to a revealed blank 'B' and all of its adjacent unrevealed squares should
	 * be revealed recursively. If an empty square 'E' with at least one adjacent
	 * mine is revealed, then change it to a digit ('1' to '8') representing the
	 * number of adjacent mines. Return the board when no more squares will be
	 * revealed.
	 */
	int[] dr = new int[] { 0, 0, -1, -1, -1, 1, 1, 1 }, dc = new int[] { -1, 1, -1, 0, 1, -1, 0, 1 };

	public char[][] updateBoard(char[][] board, int[] click) {
		int m = board.length, n = board[0].length;
		int r = click[0], c = click[1];
		if (board[r][c] == 'M') {
			board[r][c] = 'X';
		} else if (board[r][c] == 'E') {
			BFS(board, r, c, m, n);
		}
		return board;
	}

	public void BFS(char[][] board, int r, int c, int m, int n) {
		Queue<int[]> queue = new LinkedList<>();
		boolean[][] vis = new boolean[m][n];
		vis[r][c] = true;
		queue.add(new int[] { r, c });
		while (!queue.isEmpty()) {
			int[] cur = queue.poll();
			int cur_r = cur[0], cur_c = cur[1];
			int cnt = countMines(board, cur_r, cur_c, m, n);
			if (cnt > 0) {
				board[cur_r][cur_c] = String.valueOf(cnt).charAt(0);
			} else {
				board[cur_r][cur_c] = 'B';
				for (int i = 0; i < 8; i++) {
					int nr = cur_r + dr[i], nc = cur_c + dc[i];
					if (nr >= 0 && nr < m && nc >= 0 && nc < n && !vis[nr][nc]) {
						vis[nr][nc] = true;
						queue.add(new int[] { nr, nc });
					}
				}
			}
		}
	}

	public int countMines(char[][] board, int r, int c, int m, int n) {
		int cnt = 0;
		for (int i = 0; i < 8; i++) {
			int nr = r + dr[i], nc = c + dc[i];
			if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
				cnt += board[nr][nc] == 'M' ? 1 : 0;
			}
		}
		return cnt;
	}

	// K-diff Pairs in an Array
	/*
	 * Given an array of integers nums and an integer k, return the number of unique
	 * k-diff pairs in the array.
	 * 
	 * A k-diff pair is an integer pair (nums[i], nums[j]), where the following are
	 * true:
	 * 
	 * 0 <= i < j < nums.length |nums[i] - nums[j]| == k Notice that |val| denotes
	 * the absolute value of val.
	 */
	public int findPairs(int[] nums, int k) {
		HashMap<Integer, Integer> map = new HashMap<>();

		for (int n : nums) {
			map.put(n, map.getOrDefault(n, 0) + 1);
		}

		int cnt = 0;
		for (int n : map.keySet()) {
			if (k > 0 && map.containsKey(n + k) || k == 0 && map.get(n) > 1) {
				cnt++;
			}
		}
		return cnt;
	}

	// Complex Number Multiplication
	/*
	 * A complex number can be represented as a string on the form "real+imaginaryi"
	 * where:
	 * 
	 * real is the real part and is an integer in the range [-100, 100]. imaginary
	 * is the imaginary part and is an integer in the range [-100, 100]. i2 == -1.
	 * Given two complex numbers num1 and num2 as strings, return a string of the
	 * complex number that represents their multiplications.
	 */
	public String complexNumberMultiply(String num1, String num2) {
		int a = Integer.parseInt(num1.substring(0, num1.indexOf("+")));
		int b = Integer.parseInt(num1.substring(num1.indexOf("+") + 1, num1.length() - 1));
		int c = Integer.parseInt(num2.substring(0, num2.indexOf("+")));
		int d = Integer.parseInt(num2.substring(num2.indexOf("+") + 1, num2.length() - 1));
		return "" + (a * c - b * d) + "+" + (a * d + c * b) + "i";
	}

	// 01 Matrix
	/*
	 * Given an m x n binary matrix mat, return the distance of the nearest 0 for
	 * each cell.
	 * 
	 * The distance between two adjacent cells is 1.
	 */
	public int[][] updateMatrix(int[][] mat) {
		if (mat == null || mat.length == 0)
			return mat;

		Queue<int[]> q = new LinkedList<>();
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[0].length; j++) {
				if (mat[i][j] == 0)
					q.offer(new int[] { i, j });
				else
					mat[i][j] = -1;
			}
		}

		int[][] dirs = { { 1, 0 }, { 0, -1 }, { 0, 1 }, { -1, 0 } };
		while (!q.isEmpty()) {
			int[] curr = q.poll();
			for (int[] dir : dirs) {
				int r = curr[0] + dir[0];
				int c = curr[1] + dir[1];

				if (r >= 0 && c >= 0 && r < mat.length && c < mat[0].length && mat[r][c] == -1) {
					q.offer(new int[] { r, c });
					mat[r][c] = mat[curr[0]][curr[1]] + 1;
				}
			}
		}

		return mat;
	}

	// Optimal Division
	/*
	 * You are given an integer array nums. The adjacent integers in nums will
	 * perform the float division.
	 * 
	 * For example, for nums = [2,3,4], we will evaluate the expression "2/3/4".
	 * However, you can add any number of parenthesis at any position to change the
	 * priority of operations. You want to add these parentheses such the value of
	 * the expression after the evaluation is maximum.
	 * 
	 * Return the corresponding expression that has the maximum value in string
	 * format.
	 * 
	 * Note: your expression should not contain redundant parenthesis.
	 */
	public String optimalDivision(int[] nums) {
		StringBuilder sb = new StringBuilder();
		if (nums.length == 1) {
			return sb.append(nums[0]).toString();
		}
		if (nums.length == 2) {
			return sb.append(nums[0]).append("/").append(nums[1]).toString();
		}
		sb.append(nums[0]).append("/").append("(");
		for (int i = 1; i < nums.length; i++) {
			sb.append(nums[i]);
			if (i < nums.length - 1) {
				sb.append("/");
			}
		}
		sb.append(")");
		return sb.toString();
	}

	// Subarray Sum Equals K
	/*
	 * Given an array of integers nums and an integer k, return the total number of
	 * continuous subarrays whose sum equals to k.
	 */
	public int subarraySum(int[] nums, int k) {
		if (nums == null || nums.length == 0) {
			return 0;
		}
		Map<Integer, List<Integer>> sumToIndex = new HashMap<>();
		sumToIndex.put(0, new ArrayList<>(Collections.singletonList(-1)));
		int sum = 0;
		int count = 0; // answer
		for (int i = 0; i < nums.length; i++) {
			sum = sum + nums[i];
			if (sumToIndex.containsKey(sum - k)) {
				count = count + sumToIndex.get(sum - k).size();
			}
			if (!sumToIndex.containsKey(sum)) {
				sumToIndex.put(sum, new ArrayList<>());
			}
			sumToIndex.get(sum).add(i);
		}
		return count;
	}

	// Subtree of Another Tree
	/*
	 * Given the roots of two binary trees root and subRoot, return true if there is
	 * a subtree of root with the same structure and node values of subRoot and
	 * false otherwise.
	 * 
	 * A subtree of a binary tree tree is a tree that consists of a node in tree and
	 * all of this node's descendants. The tree tree could also be considered as a
	 * subtree of itself.
	 */
	public boolean isSubtree(TreeNode root, TreeNode subRoot) {
		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		queue.add(root);
		while (!queue.isEmpty()) {
			int size = queue.size();
			for (int i = 0; i < size; i++) {
				TreeNode node = queue.poll();
				if (isIdentical(node, subRoot)) {
					return true;
				}
				if (node.left != null) {
					queue.add(node.left);
				}
				if (node.right != null) {
					queue.add(node.right);
				}
			}
		}
		return false;
	}

	private boolean isIdentical(TreeNode node, TreeNode subRoot) {
		if (node == null && subRoot == null) {
			return true;
		}
		if (node == null || subRoot == null) {
			return false;
		}
		if (node.val != subRoot.val) {
			return false;
		}
		return isIdentical(node.left, subRoot.left) && isIdentical(node.right, subRoot.right);
	}

	// Shortest Unsorted Continuous Subarray
	/*
	 * Given an integer array nums, you need to find one continuous subarray that if
	 * you only sort this subarray in ascending order, then the whole array will be
	 * sorted in ascending order.
	 * 
	 * Return the shortest such subarray and output its length.
	 */
	public int findUnsortedSubarray(int[] nums) {
		int n = nums.length;
		int min = nums[n - 1];
		int max = nums[0];
		int l = -1;
		int r = -2;
		for (int i = 1; i < n; i++) {
			min = Math.min(min, nums[n - i - 1]);
			max = Math.max(max, nums[i]);
			if (nums[i] < max) {
				r = i;
			}
			if (min < nums[n - i - 1]) {
				l = n - i - 1;
			}
		}
		return r - l + 1;
	}

	// Construct String from Binary Tree
	/*
	 * Given the root of a binary tree, construct a string consisting of parenthesis
	 * and integers from a binary tree with the preorder traversal way, and return
	 * it.
	 * 
	 * Omit all the empty parenthesis pairs that do not affect the one-to-one
	 * mapping relationship between the string and the original binary tree.
	 */
	public String tree2str(TreeNode root) {
		return construct(root);
	}

	private String construct(TreeNode root) {

		StringBuilder res = new StringBuilder();

		if (root == null)
			return "";

		res.append(root.val);

		String left = construct(root.left);
		String right = construct(root.right);

		if (left.length() != 0) {
			res.append("(");
			res.append(left);
			res.append(")");
		}

		if (right.length() != 0) {
			if (left.length() == 0)
				res.append("()");
			res.append("(");
			res.append(right);
			res.append(")");
		}
		return res.toString();
	}

	// Solve the Equation
	/*
	 * Solve a given equation and return the value of 'x' in the form of a string
	 * "x=#value". The equation contains only '+', '-' operation, the variable 'x'
	 * and its coefficient. You should return "No solution" if there is no solution
	 * for the equation, or "Infinite solutions" if there are infinite solutions for
	 * the equation.
	 * 
	 * If there is exactly one solution for the equation, we ensure that the value
	 * of 'x' is an integer.
	 */
	public String solveEquation(String equation) {
		String[] arr = equation.split("=");
		int[] left = evaluate(arr[0]);
		int[] right = evaluate(arr[1]);
		int countX = left[0] - right[0];
		int countNum = left[1] - right[1];
		if (countX == 0) {
			if (countNum == 0)
				return "Infinite solutions";
			return "No solution";
		}
		int val = -1 * (countNum / countX);
		StringBuilder ans = new StringBuilder("");
		ans.append("x=");
		ans.append(val);
		return ans.toString();
	}

	public int[] evaluate(String s) {
		String[] str = s.split("(?=[-+])");
		int[] res = new int[2];
		for (String t : str) {
			if (t.equals("+x") || t.equals("x")) {
				res[0]++;
			} else if (t.equals("-x")) {
				res[0]--;
			} else if (t.contains("x")) {
				res[0] += Integer.valueOf(t.substring(0, t.indexOf("x")));
			} else {
				res[1] += Integer.valueOf(t);
			}
		}
		return res;
	}

	// Set Mismatch
	/*
	 * You have a set of integers s, which originally contains all the numbers from
	 * 1 to n. Unfortunately, due to some error, one of the numbers in s got
	 * duplicated to another number in the set, which results in repetition of one
	 * number and loss of another number.
	 * 
	 * You are given an integer array nums representing the data status of this set
	 * after the error.
	 * 
	 * Find the number that occurs twice and the number that is missing and return
	 * them in the form of an array.
	 */
	public int[] findErrorNums(int[] nums) {
		int i = 0;
		int n = nums.length;
		while (i < n) {
			if (nums[i] != nums[nums[i] - 1]) {
				swapNums(nums, i, nums[i] - 1);
			} else {
				i++;
			}
		}
		for (i = 0; i < n; i++) {
			if (nums[i] != i + 1) {
				return new int[] { nums[i], i + 1 };
			}
		}
		return new int[] {};
	}

	private void swapNums(int[] a, int i, int j) {
		int tmp = a[i];
		a[i] = a[j];
		a[j] = tmp;
	}

	// Maximum Length of Pair Chain
	/*
	 * You are given an array of n pairs pairs where pairs[i] = [lefti, righti] and
	 * lefti < righti.
	 * 
	 * A pair p2 = [c, d] follows a pair p1 = [a, b] if b < c. A chain of pairs can
	 * be formed in this fashion.
	 * 
	 * Return the length longest chain which can be formed.
	 * 
	 * You do not need to use up all the given intervals. You can select pairs in
	 * any order.
	 */
	public int findLongestChain(int[][] pairs) {
		Arrays.sort(pairs, (paira, pairb) -> (paira[0] - pairb[0]));
		TreeMap<Integer, Integer> map = new TreeMap<>();
		for (int[] pair : pairs) {
			if (map.size() == 0)
				map.put(pair[0], pair[1]);
			if (map.get(map.floorKey(pair[0])) < pair[0]) {
				map.put(pair[0], pair[1]);
			} else if (map.get(map.floorKey(pair[0])) >= pair[0] && map.get(map.floorKey(pair[0])) <= pair[1]) {
				continue;
			} else if (map.get(map.floorKey(pair[0])) > pair[1]) {
				map.remove(map.floorKey(pair[0]));
				map.put(pair[0], pair[1]);
			}
		}
		return map.size();
	}

	// Image Smoother
	/*
	 * An image smoother is a filter of the size 3 x 3 that can be applied to each
	 * cell of an image by rounding down the average of the cell and the eight
	 * surrounding cells (i.e., the average of the nine cells in the blue smoother).
	 * If one or more of the surrounding cells of a cell is not present, we do not
	 * consider it in the average (i.e., the average of the four cells in the red
	 * smoother).
	 */
	public int[][] imageSmoother(int[][] img) {
		int m = img.length, n = img[0].length;
		int[][] dirs = { { -1, 0 }, { 1, 0 }, { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 }, { 0, 1 }, { 0, -1 } };
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				int sum = img[i][j] & (0xFF), cnt = 1;
				for (int[] dir : dirs) {
					int nx = i + dir[0], ny = j + dir[1];
					if (nx < 0 || nx >= m || ny < 0 || ny >= n) {
						continue;
					}
					sum += img[nx][ny] & (0xFF);
					cnt++;
				}
				int ans = sum / cnt;
				img[i][j] |= (ans << 9);
			}
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				img[i][j] = img[i][j] >> 9;
			}
		}

		return img;
	}

	// Cut Off Trees for Golf Event
	/*
	 * You are asked to cut off all the trees in a forest for a golf event. The
	 * forest is represented as an m x n matrix. In this matrix:
	 * 
	 * 0 means the cell cannot be walked through. 1 represents an empty cell that
	 * can be walked through. A number greater than 1 represents a tree in a cell
	 * that can be walked through, and this number is the tree's height. In one
	 * step, you can walk in any of the four directions: north, east, south, and
	 * west. If you are standing in a cell with a tree, you can choose whether to
	 * cut it off.
	 * 
	 * You must cut off the trees in order from shortest to tallest. When you cut
	 * off a tree, the value at its cell becomes 1 (an empty cell).
	 * 
	 * Starting from the point (0, 0), return the minimum steps you need to walk to
	 * cut off all the trees. If you cannot cut off all the trees, return -1.
	 * 
	 * You are guaranteed that no two trees have the same height, and there is at
	 * least one tree needs to be cut off.
	 */
	public int cutOffTree(List<List<Integer>> forest) {
		int n = forest.size();
		int m = forest.get(0).size();
		int dir[][] = { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };

		PriorityQueue<Integer> minheap = new PriorityQueue<>();
		for (List<Integer> al : forest)
			for (int val : al)
				if (val > 1)
					minheap.add(val);

		int result = 0;
		int[] src = { 0, 0 };

		while (minheap.size() > 0) {
			boolean vis[][] = new boolean[n][m];
			Queue<int[]> q = new LinkedList<>();
			q.add(new int[] { src[0], src[1], 0 });
			int target = minheap.remove(), ans = -1;
			while (!q.isEmpty()) {
				int a[] = q.remove();
				if (vis[a[0]][a[1]])
					continue;
				vis[a[0]][a[1]] = true;
				if (forest.get(a[0]).get(a[1]) == target) {
					src[0] = a[0];
					src[1] = a[1];
					ans = a[2];
					break;
				}

				for (int b[] : dir) {
					int r = a[0] + b[0];
					int c = a[1] + b[1];
					if (r < 0 || r >= n || c < 0 || c >= m || vis[r][c] || forest.get(r).get(c) == 0)
						continue;
					q.add(new int[] { r, c, a[2] + 1 });
				}
			}

			if (ans == -1)
				return -1;
			result += ans;
		}
		return result;
	}

	// Baseball Game
	/*
	 * You are keeping score for a baseball game with strange rules. The game
	 * consists of several rounds, where the scores of past rounds may affect future
	 * rounds' scores.
	 * 
	 * At the beginning of the game, you start with an empty record. You are given a
	 * list of strings ops, where ops[i] is the ith operation you must apply to the
	 * record and is one of the following:
	 * 
	 * An integer x - Record a new score of x. "+" - Record a new score that is the
	 * sum of the previous two scores. It is guaranteed there will always be two
	 * previous scores. "D" - Record a new score that is double the previous score.
	 * It is guaranteed there will always be a previous score. "C" - Invalidate the
	 * previous score, removing it from the record. It is guaranteed there will
	 * always be a previous score. Return the sum of all the scores on the record.
	 */
	public int calPoints(String[] ops) {
		int sum = 0;
		Stack<Integer> s = new Stack<>();
		for (int i = 0; i < ops.length; i++) {
			if (ops[i].equals("D")) {
				int x = s.pop();
				int y = 2 * x;
				s.push(x);
				s.push(y);
			} else if (ops[i].equals("+")) {
				int x = s.pop();
				int y = s.pop();
				int z = x + y;
				s.push(y);
				s.push(x);
				s.push(z);
			} else if (ops[i].equals("C")) {
				s.pop();
			} else {
				int a = Integer.parseInt(ops[i]);
				s.push(a);
			}
		}
		while (!s.isEmpty()) {
			sum += s.pop();
		}
		return sum;
	}

	// Top K Frequent Words
	/*
	 * Given an array of strings words and an integer k, return the k most frequent
	 * strings.
	 * 
	 * Return the answer sorted by the frequency from highest to lowest. Sort the
	 * words with the same frequency by their lexicographical order.
	 */
	public List<String> topKFrequent(String[] words, int k) {
		List<String> result = new ArrayList<>();
		Map<String, Integer> map = new HashMap<String, Integer>();
		for (String s : words)
			map.put(s, map.getOrDefault(s, 0) + 1);
		PriorityQueue<Map.Entry<String, Integer>> q = new PriorityQueue<>(
				(a, b) -> a.getValue() != b.getValue() ? -1 * a.getValue().compareTo(b.getValue())
						: a.getKey().compareTo(b.getKey()));
		for (Map.Entry<String, Integer> m : map.entrySet())
			q.add(m);
		for (int i = 0; i < k; ++i)
			result.add(q.poll().getKey());
		return result;
	}

	// Find Pivot Index
	/*
	 * Given an array of integers nums, calculate the pivot index of this array.
	 * 
	 * The pivot index is the index where the sum of all the numbers strictly to the
	 * left of the index is equal to the sum of all the numbers strictly to the
	 * index's right.
	 * 
	 * If the index is on the left edge of the array, then the left sum is 0 because
	 * there are no elements to the left. This also applies to the right edge of the
	 * array.
	 * 
	 * Return the leftmost pivot index. If no such index exists, return -1.
	 */
	public int pivotIndex(int[] nums) {
		int total = 0;
		for (int num : nums) {
			total += num;
		}

		int leftSum = 0;
		for (int i = 0; i < nums.length; i++) {
			int rigthSum = total - nums[i] - leftSum;
			if (leftSum == rigthSum) {
				return i;
			}
			leftSum += nums[i];
		}
		return -1;
	}

	// Split Linked List in Parts
	/*
	 * Given the head of a singly linked list and an integer k, split the linked
	 * list into k consecutive linked list parts.
	 * 
	 * The length of each part should be as equal as possible: no two parts should
	 * have a size differing by more than one. This may lead to some parts being
	 * null.
	 * 
	 * The parts should be in the order of occurrence in the input list, and parts
	 * occurring earlier should always have a size greater than or equal to parts
	 * occurring later.
	 * 
	 * Return an array of the k parts.
	 */
	public ListNode[] splitListToParts(ListNode head, int k) {
		int l = len(head);
		ListNode[] ans = new ListNode[k];
		ListNode cur = head;
		int mod = l % k;
		int j = 0;
		while (cur != null) {
			int size = l / k;
			if (mod > 0) {
				mod--;
				size++;
			}
			ans[j++] = cur;
			for (int i = 0; i < size - 1 && cur != null; i++) {
				cur = cur.next;
			}
			if (cur != null) {
				ListNode nxt = cur.next;
				cur.next = null;
				cur = nxt;
			}
		}
		return ans;
	}

	int len(ListNode root) {
		int ans = 0;
		while (root != null) {
			ans++;
			root = root.next;
		}
		return ans;
	}

	// Monotone Increasing Digits
	/*
	 * An integer has monotone increasing digits if and only if each pair of
	 * adjacent digits x and y satisfy x <= y.
	 * 
	 * Given an integer n, return the largest number that is less than or equal to n
	 * with monotone increasing digits.
	 */
	public int monotoneIncreasingDigits(int n) {
		for (int i = 10; n / i > 0; i *= 10) {
			int digit = (n / i) % 10;
			int endnum = n % i;
			int firstendnum = endnum * 10 / i;
			if (digit > firstendnum) {
				n -= endnum + 1;
			}
		}
		return (n);
	}

	// Prime Number of Set Bits in Binary Representation
	/*
	 * Given two integers left and right, return the count of numbers in the
	 * inclusive range [left, right] having a prime number of set bits in their
	 * binary representation.
	 * 
	 * Recall that the number of set bits an integer has is the number of 1's
	 * present when written in binary.
	 * 
	 * For example, 21 written in binary is 10101, which has 3 set bits.
	 */
	public int countPrimeSetBits(int left, int right) {
		int countNoOfPrimes = 0;
		for (int i = left; i <= right; i++) {
			int temp = i;
			int count = 0;
			while (temp > 0) {
				if ((temp & 1) == 1)
					count++;
				temp = temp >> 1;
			}
			if (checkPrime(count))
				countNoOfPrimes++;
		}
		return countNoOfPrimes;
	}

	public static boolean checkPrime(int a) {
		if (a <= 1)
			return false;
		for (int i = 2; i * i <= a; i++) {
			if (a % i == 0) {
				return false;
			}
		}
		return true;
	}

	// Partition Labels
	/*
	 * You are given a string s. We want to partition the string into as many parts
	 * as possible so that each letter appears in at most one part.
	 * 
	 * Note that the partition is done so that after concatenating all the parts in
	 * order, the resultant string should be s.
	 * 
	 * Return a list of integers representing the size of these parts.
	 */
	public List<Integer> partitionLabels(String s) {
		HashMap<Character, Integer> map = new HashMap<>();
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i < s.length(); i++) {
			map.put(s.charAt(i), i);

		}
		int prev = -1;
		int max = 0;
		for (int i = 0; i < s.length(); i++) {
			max = Math.max(max, map.get(s.charAt(i)));
			if (max == i) {
				list.add(max - prev);
				prev = max;

			}
		}
		return list;
	}

	// Reorganize String
	/*
	 * Given a string s, rearrange the characters of s so that any two adjacent
	 * characters are not the same.
	 * 
	 * Return any possible rearrangement of s or return "" if not possible.
	 */
	public String reorganizeString(String s) {
		int alpha[] = new int[26];
		for (int i = 0; i < s.length(); i++)
			alpha[s.charAt(i) - 'a']++;
		int maxIdx = 0;
		char charMax = 'A';
		for (int i = 0; i < 26; i++) {
			if (alpha[i] >= alpha[maxIdx]) {
				maxIdx = i;
				charMax = (char) (i + 'a');
			}
		}
		if (alpha[maxIdx] > (s.length() + 1) / 2)
			return "";
		char[] charArray = new char[s.length()];
		int idx = 0;
		while (idx < charArray.length) {
			charArray[idx] = charMax;
			alpha[maxIdx]--;
			idx += 2;
			if (alpha[maxIdx] == 0)
				break;
		}
		for (int i = 0; i < 26; i++) {
			while (alpha[i] > 0) {
				if (idx >= charArray.length) {
					idx = 1;
				}
				charArray[idx] = (char) (i + 'a');
				alpha[i]--;
				idx += 2;
			}
		}
		return String.valueOf(charArray);
	}

	// Global and Local Inversions
	/*
	 * You are given an integer array nums of length n which represents a
	 * permutation of all the integers in the range [0, n - 1].
	 * 
	 * The number of global inversions is the number of the different pairs (i, j)
	 * where:
	 * 
	 * 0 <= i < j < n nums[i] > nums[j] The number of local inversions is the number
	 * of indices i where:
	 * 
	 * 0 <= i < n - 1 nums[i] > nums[i + 1] Return true if the number of global
	 * inversions is equal to the number of local inversions.
	 */
	public boolean isIdealPermutation(int[] nums) {
		int[] rightMinArray = new int[nums.length];
		int minVal = Integer.MAX_VALUE;
		for (int i = nums.length - 1; i >= 0; i--) {
			rightMinArray[i] = Math.min(minVal, nums[i]);
			minVal = nums[i] < minVal ? nums[i] : minVal;
		}
		for (int i = 0; i < nums.length - 2; i++) {
			if (rightMinArray[i + 2] < nums[i]) {
				return false;
			}
		}
		return true;
	}

	// Unique Email Addresses
	/*
	 * Every valid email consists of a local name and a domain name, separated by
	 * the '@' sign. Besides lowercase letters, the email may contain one or more
	 * '.' or '+'.
	 * 
	 * For example, in "alice@leetcode.com", "alice" is the local name, and
	 * "leetcode.com" is the domain name. If you add periods '.' between some
	 * characters in the local name part of an email address, mail sent there will
	 * be forwarded to the same address without dots in the local name. Note that
	 * this rule does not apply to domain names.
	 * 
	 * For example, "alice.z@leetcode.com" and "alicez@leetcode.com" forward to the
	 * same email address. If you add a plus '+' in the local name, everything after
	 * the first plus sign will be ignored. This allows certain emails to be
	 * filtered. Note that this rule does not apply to domain names.
	 * 
	 * For example, "m.y+name@email.com" will be forwarded to "my@email.com". It is
	 * possible to use both of these rules at the same time.
	 * 
	 * Given an array of strings emails where we send one email to each emails[i],
	 * return the number of different addresses that actually receive mails.
	 */
	public int numUniqueEmails(String[] emails) {
		return (int) Stream.of(emails)
				.map(str -> str.split("@")[0].split("\\+")[0].replace(".", "") + "@" + str.split("@")[1]).distinct()
				.count();
	}

	// Knight Dialer
	/*
	 * The chess knight has a unique movement, it may move two squares vertically
	 * and one square horizontally, or two squares horizontally and one square
	 * vertically (with both forming the shape of an L). The possible movements of
	 * chess knight are shown in this diagaram:
	 * 
	 * A chess knight can move as indicated in the chess diagram below:
	 * 
	 * 
	 * We have a chess knight and a phone pad as shown below, the knight can only
	 * stand on a numeric cell (i.e. blue cell).
	 * 
	 * 
	 * Given an integer n, return how many distinct phone numbers of length n we can
	 * dial.
	 * 
	 * You are allowed to place the knight on any numeric cell initially and then
	 * you should perform n - 1 jumps to dial a number of length n. All jumps should
	 * be valid knight jumps.
	 * 
	 * As the answer may be very large, return the answer modulo 109 + 7.
	 */

	static int mod = 1000_000_007;
	static int[][] map = new int[10][];
	static List<int[]> memo = new ArrayList<>();
	static {
		map[0] = new int[] { 4, 6 };
		map[1] = new int[] { 6, 8 };
		map[2] = new int[] { 7, 9 };
		map[3] = new int[] { 4, 8 };
		map[4] = new int[] { 3, 9, 0 };
		map[5] = new int[0];
		map[6] = new int[] { 1, 7, 0 };
		map[7] = new int[] { 2, 6 };
		map[8] = new int[] { 1, 3 };
		map[9] = new int[] { 2, 4 };
		memo.add(new int[] { 1, 1, 1, 1, 1, 0, 1, 1, 1, 1 });
	}

	public int knightDialer(int n) {
		if (n == 1)
			return 10;
		while (memo.size() < n) {
			int[] cur = memo.get(memo.size() - 1);
			int[] next = new int[10];
			for (int i = 0; i < 10; i++) {
				for (int d : map[i]) {
					next[d] = (next[d] + cur[i]) % mod;
				}
			}
			memo.add(next);
		}
		int sum = 0;
		for (int x : memo.get(n - 1)) {
			sum = (sum + x) % mod;
		}
		return sum;
	}

	// Reorder Data in Log Files
	/*
	 * You are given an array of logs. Each log is a space-delimited string of
	 * words, where the first word is the identifier.
	 * 
	 * There are two types of logs:
	 * 
	 * Letter-logs: All words (except the identifier) consist of lowercase English
	 * letters. Digit-logs: All words (except the identifier) consist of digits.
	 * Reorder these logs so that:
	 * 
	 * The letter-logs come before all digit-logs. The letter-logs are sorted
	 * lexicographically by their contents. If their contents are the same, then
	 * sort them lexicographically by their identifiers. The digit-logs maintain
	 * their relative ordering. Return the final order of the logs.
	 */
	public String[] reorderLogFiles(String[] logs) {
		String[] res = new String[logs.length];
		List<String> letterLogs = new ArrayList<>();
		List<String> digitLogs = new ArrayList<>();
		for (String log : logs) {
			if (Character.isLetter(log.charAt(log.indexOf(" ") + 1)))
				letterLogs.add(log);
			else
				digitLogs.add(log);
		}
		letterLogs.sort((o1, o2) -> {
			int cmp = o1.substring(o1.indexOf(" ") + 1).compareTo(o2.substring(o2.indexOf(" ") + 1));
			if (cmp == 0)
				return o1.compareTo(o2);
			return cmp;
		});
		int i = 0;
		for (String log : letterLogs)
			res[i++] = log;
		for (String log : digitLogs)
			res[i++] = log;
		return res;
	}

	// Prison Cells After N Days
	/*
	 * There are 8 prison cells in a row and each cell is either occupied or vacant.
	 * 
	 * Each day, whether the cell is occupied or vacant changes according to the
	 * following rules:
	 * 
	 * If a cell has two adjacent neighbors that are both occupied or both vacant,
	 * then the cell becomes occupied. Otherwise, it becomes vacant. Note that
	 * because the prison is a row, the first and the last cells in the row can't
	 * have two adjacent neighbors.
	 * 
	 * You are given an integer array cells where cells[i] == 1 if the ith cell is
	 * occupied and cells[i] == 0 if the ith cell is vacant, and you are given an
	 * integer n.
	 * 
	 * Return the state of the prison after n days (i.e., n such changes described
	 * above).
	 */
	public int[] prisonAfterNDays(int[] cells, int n) {
		Set<String> st = new HashSet<>();
		while (n-- > 0) {
			int prev = cells[0];
			for (int i = 1; i < cells.length - 1; i++) {
				if ((prev ^ cells[i + 1]) == 1) {
					prev = cells[i];
					cells[i] = 0;
				} else {
					prev = cells[i];
					cells[i] = 1;
				}
			}
			cells[0] = cells[cells.length - 1] = 0;
			String str = Arrays.toString(cells);
			if (st.contains(str))
				n %= st.size();
			else
				st.add(str);
		}
		return cells;
	}

	// K Closest Points to Origin
	/*
	 * Given an array of points where points[i] = [xi, yi] represents a point on the
	 * X-Y plane and an integer k, return the k closest points to the origin (0, 0).
	 * 
	 * The distance between two points on the X-Y plane is the Euclidean distance
	 * (i.e., sqrt((x1 - x2)2 + (y1 - y2)2)).
	 * 
	 * You may return the answer in any order. The answer is guaranteed to be unique
	 * (except for the order that it is in).
	 */
	public int[][] kClosest(int[][] points, int k) {
		Arrays.sort(points, (a, b) -> (a[0] * a[0] + a[1] * a[1]) - (b[0] * b[0] + b[1] * b[1]));
		int[][] ret = new int[k][];
		for (int i = 0; i < k; i++) {
			ret[i] = points[i];
		}
		return ret;
	}

	// Subarrays with K Different Integers
	/*
	 * Given an integer array nums and an integer k, return the number of good
	 * subarrays of nums.
	 * 
	 * A good array is an array where the number of different integers in that array
	 * is exactly k.
	 * 
	 * For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3. A subarray is
	 * a contiguous part of an array.
	 */
	public int subarraysWithKDistinct(int[] nums, int k) {
		int res = 0, prefix = 0;
		int[] cnt = new int[nums.length + 1];
		for (int i = 0, j = 0, unique_count = 0; i < nums.length; i++) {
			if (cnt[nums[i]]++ == 0)
				unique_count++;
			if (unique_count > k) {
				--cnt[nums[j++]];
				prefix = 0;
				unique_count--;
			}
			while (cnt[nums[j]] > 1) {
				++prefix;
				--cnt[nums[j++]];
			}
			if (unique_count == k)
				res += prefix + 1;
		}
		return res;
	}

	// Rotting Oranges
	/*
	 * You are given an m x n grid where each cell can have one of three values:
	 * 
	 * 0 representing an empty cell, 1 representing a fresh orange, or 2
	 * representing a rotten orange. Every minute, any fresh orange that is
	 * 4-directionally adjacent to a rotten orange becomes rotten.
	 * 
	 * Return the minimum number of minutes that must elapse until no cell has a
	 * fresh orange. If this is impossible, return -1.
	 */
	private static final int[][] dirs = new int[][] { { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };

	public int orangesRotting(int[][] grid) {
		int m = grid.length;
		if (m == 0)
			return 0;
		int freshCount = 0;
		Queue<int[]> rotten = new LinkedList<>();
		for (int i = 0; i < grid.length; ++i) {
			for (int j = 0; j < grid[0].length; ++j) {
				if (grid[i][j] == 1)
					freshCount += 1;
				if (grid[i][j] == 2) {
					rotten.add(new int[] { i, j });
				}
			}
		}
		if (freshCount == 0)
			return 0;
		int minutePassed = 0;
		while (!rotten.isEmpty()) {
			minutePassed++;
			int numRotten = rotten.size();
			for (int i = 0; i < numRotten; ++i) {
				int[] curr = rotten.poll();
				int row = curr[0], col = curr[1];
				for (int[] dir : dirs) {
					int newr = row + dir[0];
					int newc = col + dir[1];
					if (newr < 0 || newr > grid.length - 1 || newc < 0 || newc > grid[0].length - 1
							|| grid[newr][newc] != 1) {
						continue;
					}
					grid[newr][newc] = 2;
					freshCount--;
					rotten.add(new int[] { newr, newc });
				}
			}
		}
		if (freshCount != 0)
			return -1;
		return minutePassed - 1;
	}

	// Binary Search Tree to Greater Sum Tree
	/*
	 * Given the root of a Binary Search Tree (BST), convert it to a Greater Tree
	 * such that every key of the original BST is changed to the original key plus
	 * the sum of all keys greater than the original key in BST.
	 * 
	 * As a reminder, a binary search tree is a tree that satisfies these
	 * constraints:
	 * 
	 * The left subtree of a node contains only nodes with keys less than the node's
	 * key. The right subtree of a node contains only nodes with keys greater than
	 * the node's key. Both the left and right subtrees must also be binary search
	 * trees.
	 */
	public TreeNode bstToGst(TreeNode root) {
		TreeNode res = root;
		int sum = dfs(root);
		Stack<TreeNode> stack = new Stack<>();
		int prev = 0;
		while (!stack.isEmpty() || root != null) {
			while (root != null) {
				stack.push(root);
				root = root.left;
			}
			root = stack.pop();
			int temp = root.val;
			root.val = sum - prev;
			prev = temp;
			sum = root.val;

			root = root.right;
		}
		return res;
	}

	private int dfs(TreeNode root) {
		if (root == null)
			return 0;
		return root.val + dfs(root.left) + dfs(root.right);
	}

	// Distant Barcodes
	/*
	 * In a warehouse, there is a row of barcodes, where the ith barcode is
	 * barcodes[i].
	 * 
	 * Rearrange the barcodes so that no two adjacent barcodes are equal. You may
	 * return any answer, and it is guaranteed an answer exists.
	 */
	public int[] rearrangeBarcodes(int[] barcodes) {
		if (barcodes.length <= 2)
			return barcodes;
		HashMap<Integer, Integer> freq = new HashMap<>();
		int max = barcodes[0];
		for (int i : barcodes) {
			freq.put(i, freq.getOrDefault(i, 0) + 1);
			max = freq.get(i) > freq.get(max) ? i : max; // Finds the most frequent element
		}
		int p1 = 0;
		int max_f = freq.get(max);
		for (p1 = 0; p1 < barcodes.length && max_f > 0; p1 += 2, max_f--)
			barcodes[p1] = max;
		for (int i : freq.keySet()) {
			if (i == max)
				continue;
			int size = freq.get(i);
			for (int j = 0; j < size; j++) {
				if (p1 == barcodes.length || p1 == barcodes.length + 1)
					p1 = 1;
				barcodes[p1] = i;
				p1 += 2;
			}
		}
		return barcodes;
	}

	// Sum of Nodes with Even-Valued Grandparent
	/*
	 * Given the root of a binary tree, return the sum of values of nodes with an
	 * even-valued grandparent. If there are no nodes with an even-valued
	 * grandparent, return 0.
	 * 
	 * A grandparent of a node is the parent of its parent if it exists.
	 */
	public int sumEvenGrandparent(TreeNode root) {
		int sum[] = { 0 };
		findSum(root, sum);
		return sum[0];
	}

	public void findSum(TreeNode root, int sum[]) {
		if (root == null)
			return;
		if (root.val % 2 == 0) {
			if (root.left != null) {
				if (root.left.left != null)
					sum[0] += root.left.left.val;
				if (root.left.right != null)
					sum[0] += root.left.right.val;
			}
			if (root.right != null) {
				if (root.right.left != null)
					sum[0] += root.right.left.val;
				if (root.right.right != null)
					sum[0] += root.right.right.val;
			}

		}
		findSum(root.left, sum);
		findSum(root.right, sum);
	}

	// Number of Dice Rolls With Target Sum
	/*
	 * You have n dice and each die has k faces numbered from 1 to k.
	 * 
	 * Given three integers n, k, and target, return the number of possible ways
	 * (out of the kn total ways) to roll the dice so the sum of the face-up numbers
	 * equals target. Since the answer may be too large, return it modulo 109 + 7.
	 */
	private static final int MOD = (int) (1E9 + 7);

	public int numRollsToTarget(int n, int k, int target) {
		if (n * k < target)
			return 0;
		Integer[][] memo = new Integer[n + 1][target + 1];
		return countRolls(n, k, target, memo);
	}

	private int countRolls(int n, int k, int target, Integer[][] memo) {
		if (target < 0 || n < 0) {
			return 0;
		}
		if (target == 0) {
			return memo[n][target] = (n == 0) ? 1 : 0;
		}
		if (memo[n][target] != null) {
			return memo[n][target];
		}
		int sum = 0;
		for (int i = 1; i <= k && target - i >= n - 1; i++) {
			sum = (sum + countRolls(n - 1, k, target - i, memo)) % MOD;
		}
		return memo[n][target] = sum;
	}

	// Critical Connections in a Network
	/*
	 * There are n servers numbered from 0 to n - 1 connected by undirected
	 * server-to-server connections forming a network where connections[i] = [ai,
	 * bi] represents a connection between servers ai and bi. Any server can reach
	 * other servers directly or indirectly through the network.
	 * 
	 * A critical connection is a connection that, if removed, will make some
	 * servers unable to reach some other server.
	 * 
	 * Return all critical connections in the network in any order.
	 */
	public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
		List<List<Integer>> res = new ArrayList<>();
		List<List<Integer>> graph = new ArrayList<List<Integer>>();
		for (int i = 0; i < n; i++) {
			graph.add(new ArrayList<>());
		}
		for (List<Integer> edge : connections) {
			graph.get(edge.get(0)).add(edge.get(1));
			graph.get(edge.get(1)).add(edge.get(0));
		}
		int[] low = new int[n];
		int[] times = new int[n];
		tarjan(0, 1, -1, times, low, new HashSet<Integer>(), res, graph);
		return res;
	}

	public void tarjan(int node, int time, int pre, int[] times, int[] low, HashSet<Integer> visited,
			List<List<Integer>> res, List<List<Integer>> graph) {
		visited.add(node);
		times[node] = time;
		low[node] = time;
		for (int next : graph.get(node)) {
			if (!visited.contains(next)) {
				tarjan(next, time + 1, node, times, low, visited, res, graph);
				low[node] = Math.min(low[node], low[next]);
			} else if (next != pre) {
				low[node] = Math.min(low[node], low[next]);
			}

			if (time < low[next]) {
				List<Integer> temp = new ArrayList<>();
				temp.add(node);
				temp.add(next);
				res.add(temp);
			}
		}
	}

	// Search Suggestions System
	/*
	 * You are given an array of strings products and a string searchWord.
	 * 
	 * Design a system that suggests at most three product names from products after
	 * each character of searchWord is typed. Suggested products should have common
	 * prefix with searchWord. If there are more than three products with a common
	 * prefix return the three lexicographically minimums products.
	 * 
	 * Return a list of lists of the suggested products after each character of
	 * searchWord is typed.
	 */
	public List<List<String>> suggestedProducts(String[] products, String searchWord) {
		SearchNode root = new SearchNode();
		for (String repo : products) {
			addWord(repo, root);
		}
		String searching = "";
		List<List<String>> result = new ArrayList<>();
		for (int i = 0; i < searchWord.length(); i++) {
			searching += Character.toString(searchWord.charAt(i));
			result.add(search(searching, root, new ArrayList<>()));
		}
		return result;
	}

	public static void addWord(String s, SearchNode root) {
		SearchNode current = root;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (current.next[c - 'a'] == null) {
				current.next[c - 'a'] = new SearchNode();
			}
			current.wordCount++;
			current = current.next[c - 'a'];
		}
		current.word = s;
		current.end = true;
	}

	public static List<String> search(String s, SearchNode root, List<String> output) {
		SearchNode current = root;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (current.next[c - 'a'] == null) {
				return output;
			}
			current = current.next[c - 'a'];
		}
		dfs(current, output);
		return output;
	}

	// Number of Steps to Reduce a Number to Zero
	/*
	 * Given an integer num, return the number of steps to reduce it to zero.
	 * 
	 * In one step, if the current number is even, you have to divide it by 2,
	 * otherwise, you have to subtract 1 from it.
	 */
	public int numberOfSteps(int num) {
		int b = steps(num, 0);
		return b;
	}

	int steps(int a, int steps) {
		if (a == 0) {
			return steps;
		}
		if (a % 2 == 0) {
			return steps(a / 2, steps + 1);
		} else {
			return steps(a - 1, steps + 1);
		}
	}

	// Stone Game III
	/*
	 * Alice and Bob continue their games with piles of stones. There are several
	 * stones arranged in a row, and each stone has an associated value which is an
	 * integer given in the array stoneValue.
	 * 
	 * Alice and Bob take turns, with Alice starting first. On each player's turn,
	 * that player can take 1, 2, or 3 stones from the first remaining stones in the
	 * row.
	 * 
	 * The score of each player is the sum of the values of the stones taken. The
	 * score of each player is 0 initially.
	 * 
	 * The objective of the game is to end with the highest score, and the winner is
	 * the player with the highest score and there could be a tie. The game
	 * continues until all the stones have been taken.
	 * 
	 * Assume Alice and Bob play optimally.
	 * 
	 * Return "Alice" if Alice will win, "Bob" if Bob will win, or "Tie" if they
	 * will end the game with the same score.
	 */
	public String stoneGameIII(int[] stoneValue) {
		int n = stoneValue.length;
		int store[] = new int[n + 1];
		int dp[] = new int[n + 1];
		for (int i = 1; i <= n; i++) {
			store[i] = stoneValue[i - 1] + store[i - 1];
		}
		for (int i = n - 1; i >= 0; i--) {
			int max = Integer.MIN_VALUE, s = 0;
			for (int j = i, p = 0; j < n && p < 3; j++, p++) {
				s = s + stoneValue[j];
				max = Math.max(max, s + store[n] - store[j + 1] - dp[j + 1]);
			}
			dp[i] = max;
		}
		if (dp[0] > store[n] - dp[0])
			return "Alice";
		if (dp[0] < store[n] - dp[0])
			return "Bob";
		return "Tie";
	}

	public static void dfs(SearchNode current, List<String> output) {
		if (current == null) {
			return;
		}
		if (output.size() >= 3) {
			return;
		}
		if (current.end) {
			output.add(current.word);
		}
		for (int i = 0; i < 26; i++) {
			if (current.next[i] != null) {
				dfs(current.next[i], output);
			}
		}
	}

	public static void main(String[] args) {
		LeetCodeTest test = new LeetCodeTest();
		System.out.println(test.letterCombinations("23"));
		System.out.println(test.letterCombinations("2"));

		int[] nums = { 2, 7, 11, 15 };
		int[] result = test.twoSum(nums, 9);
		System.out.printf("%d %d%n", result[0], result[1]);

		int[] nums2 = { 3, 2, 4 };
		int[] result2 = test.twoSum(nums2, 6);
		System.out.printf("%d %d%n", result2[0], result2[1]);

		ListNode node1 = new ListNode(2);
		node1.next = new ListNode(4);
		node1.next.next = new ListNode(3);

		ListNode node2 = new ListNode(5);
		node2.next = new ListNode(6);
		node2.next.next = new ListNode(4);

		printLinkedList(test.addTwoNumbers(node1, node2));
		System.out.println();

		System.out.println(test.lengthOfLongestSubstring("abcabcbb"));

		int[] median1 = { 1, 3 };
		int[] median2 = { 2 };
		System.out.println(test.findMedianSortedArrays(median1, median2));

		System.out.println(test.longestPalindrome("babad"));
		System.out.println(test.longestPalindrome("cbbd"));

		System.out.println(test.convert("PAYPALISHIRING", 3));

		System.out.println(test.myAtoi("42"));
		System.out.println(test.myAtoi("-42"));
		System.out.println(test.myAtoi("  -42"));
		System.out.println(test.myAtoi("  +42"));
		System.out.println(test.myAtoi("4193 test"));
		System.out.println(test.myAtoi("4"));
		System.out.println(test.myAtoi(" "));
		System.out.println(test.myAtoi("  "));

		int[] sumArray = { -1, 0, 1, 2, -1, -4 };
		System.out.println(test.threeSum(sumArray));

		System.out.println(test.generateParenthesis(3));
		System.out.println(test.generateParenthesis(2));

		int[] height = { 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1 };
		System.out.println(test.trap(height));
		int[] height2 = { 4, 2, 0, 3, 2, 5 };
		System.out.println(test.trap(height2));

		int[] permutate = { 1, 2, 3 };
		System.out.println(test.permute(permutate));
		int[] permutate2 = { 0, 1 };
		System.out.println(test.permute(permutate2));

		int[][] matrix = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
		test.rotate(matrix);
		for (int[] matrixValues : matrix) {
			System.out.println(Arrays.toString(matrixValues));
		}

		String[] anagrams = { "eat", "tea", "tan", "ate", "nat", "bat" };
		System.out.println(test.groupAnagrams(anagrams));
		System.out.println(test.groupAnagrams(anagrams));
		System.out.println(test.groupAnagrams(anagrams));

		int[][] intervals = { { 1, 3 }, { 2, 6 }, { 8, 10 }, { 15, 18 } };
		Arrays.stream(test.merge(intervals)).map(Arrays::toString).forEach(System.out::println);

		Arrays.stream(test.generateMatrix(3)).map(Arrays::toString).forEach(System.out::println);

		System.out.println(test.minPathSum(intervals));

		System.out.println(test.minDistance("horse", "ros"));
		System.out.println(test.minDistance("intention", "execution"));

		int[][] zeroMatrix = { { 1, 1, 1 }, { 1, 0, 1 }, { 1, 1, 0 }, { 1, 1, 1 } };
		test.setZeroes(zeroMatrix);
		Arrays.stream(zeroMatrix).map(Arrays::toString).forEach(System.out::println);

		int[] colors = { 2, 0, 2, 1, 1, 0 };
		test.sortColors(colors);
		System.out.println(Arrays.toString(colors));

		System.out.println(test.minWindow("ADOBECODEBANC", "ABC"));
		System.out.println(test.minWindow("a", "aa"));

		System.out.println(test.subsets(nums));

		System.out.println(test.grayCode(3));

		TreeNode node = new TreeNode(2, new TreeNode(1), new TreeNode(3));
		TreeNode node3 = new TreeNode(5);
		node3.left = new TreeNode(4);
		node3.right = new TreeNode(6);
		node3.right.left = new TreeNode(3);
		node3.right.right = new TreeNode(7);

		System.out.println(test.isValidBST(node));
		System.out.println(test.isValidBST(node3));

		System.out.println(test.getRow(0));
		System.out.println(test.getRow(1));
		System.out.println(test.getRow(3));
		System.out.println(test.getRow(5));
		System.out.println(test.getRow(4));

		int[] prices = { 7, 1, 5, 3, 6, 4 };
		System.out.println(test.maxProfit(prices));
		int[] prices2 = { 7, 6, 4, 3, 1 };
		System.out.println(test.maxProfit(prices2));

		System.out.println(test.sumNumbers(node3));

		System.out.println(test.wordBreak("leetcode", Arrays.asList("leet", "code")));
		System.out.println(test.wordBreak("applepenapple", Arrays.asList("apple", "pen")));
		System.out.println(test.wordBreak("catsandog", Arrays.asList("cats", "dog", "sand", "and", "cat")));

		int[] rotate = { 1, 2, 3, 4, 5, 6, 7 };
		test.rotate(rotate, 3);
		System.out.println(Arrays.toString(rotate));

		char[][] islands = { { '1', '1', '1', '1', '0' }, { '1', '1', '0', '1', '0' }, { '1', '1', '0', '0', '0' },
				{ '0', '0', '0', '0', '0' } };
		System.out.println(test.numIslands(islands));
		char[][] islands2 = { { '1', '1', '0', '0', '0' }, { '1', '1', '0', '0', '0' }, { '0', '0', '1', '0', '0' },
				{ '0', '0', '0', '1', '1' } };
		System.out.println(test.numIslands(islands2));

		System.out.println(test.isHappy(19));
		System.out.println(test.isHappy(2));

		System.out.println(test.countPrimes(10));
		System.out.println(test.countPrimes(0));
		System.out.println(test.countPrimes(1));
		System.out.println(test.countPrimes(13));

		ListNode reverse = new ListNode(1);
		reverse.next = new ListNode(2);
		reverse.next.next = new ListNode(3);
		reverse.next.next.next = new ListNode(4);
		reverse.next.next.next.next = new ListNode(5);

		printLinkedList(reverse);
		System.out.println();
		printLinkedList(test.reverseList(reverse));
		System.out.println();

		char[][] board = { { 'o', 'a', 'a', 'n' }, { 'e', 't', 'a', 'e' }, { 'i', 'h', 'k', 'r' },
				{ 'i', 'f', 'l', 'v' } };
		String[] words = { "oath", "pea", "eat", "rain" };
		System.out.println(test.findWords(board, words));

		int[] kthlargest = { 3, 2, 1, 5, 6, 4 };
		System.out.println(test.findKthLargest(kthlargest, 2));
		int[] kthlargest2 = { 3, 2, 3, 1, 2, 4, 5, 5, 6 };
		System.out.println(test.findKthLargest(kthlargest2, 4));

		System.out.println(test.calculate("1+1"));
		System.out.println(test.calculate("2-1"));
		System.out.println(test.calculate("(1+(4+5+2)-3)+(6+8)"));

		int[] product = { 1, 2, 3, 4 };
		System.out.println(Arrays.toString(test.productExceptSelf(product)));

		int[] windows = { 1, 3, -1, -3, 5, 3, 6, 7 };
		System.out.println(Arrays.toString(test.maxSlidingWindow(windows, 3)));

		System.out.println(test.isAnagram("cat", "act"));
		System.out.println(test.isAnagram("bat", "tac"));

		int[] frequency = { 1, 1, 1, 2, 2, 3 };
		System.out.println(Arrays.toString(test.topKFrequent(frequency, 2)));
		System.out.println(Arrays.toString(test.topKFrequent(frequency, 3)));

		System.out.println(test.firstUniqChar("leetcode"));
		System.out.println(test.firstUniqChar("loveleetcode"));
		System.out.println(test.firstUniqChar("aabb"));

		int[] rotationSum = { 4, 3, 2, 6 };
		System.out.println(test.maxRotateFunction(rotationSum));
		int[] rotationSum2 = { 1, 2, 3, 4, 5 };
		System.out.println(test.maxRotateFunction(rotationSum2));

		int[] thirdMax = { 3, 2, 1 };
		System.out.println(test.thirdMax(thirdMax));
		int[] thirdMax2 = { 2, 1 };
		System.out.println(test.thirdMax(thirdMax2));
		int[] thirdMax3 = { 2, 2, 3, 1 };
		System.out.println(test.thirdMax(thirdMax3));

		char[][] battleShips = { { 'X', '.', '.', 'X' }, { '.', '.', '.', 'X' }, { '.', '.', '.', 'X' },
				{ '.', '.', '.', '.' } };
		System.out.println(test.countBattleships(battleShips));

		System.out.println(test.findAnagrams("cbaebabacd", "abc"));
		System.out.println(test.findAnagrams("abab", "ab"));

		char[] compress = { 'a', 'a', 'b', 'b', 'c', 'c', 'c' };
		System.out.println(Arrays.toString(Arrays.copyOfRange(compress, 0, test.compress(compress))));

		System.out.println(test.frequencySort("tree"));
		System.out.println(test.frequencySort("cccaaa"));
		System.out.println(test.frequencySort("Aabb"));

		System.out.println(test.repeatedSubstringPattern("abab"));
		System.out.println(test.repeatedSubstringPattern("abc"));
		System.out.println(test.repeatedSubstringPattern("abcabcabcabc"));

		String[] concatenated = { "cat", "cats", "catsdogcats", "dog", "dogcatsdog", "hippopotamuses", "rat",
				"ratcatdogcat" };
		System.out.println(test.findAllConcatenatedWordsInADict(concatenated));

		System.out.println(test.longestPalindromeSubseq("bbbab"));
		System.out.println(test.longestPalindromeSubseq("cbbd"));

		int[] machines = { 1, 0, 5 };
		System.out.println(test.findMinMoves(machines));
		int[] machines2 = { 0, 3, 0 };
		System.out.println(test.findMinMoves(machines2));
		int[] machines3 = { 0, 2, 0 };
		System.out.println(test.findMinMoves(machines3));

		int[] pairs = { 3, 1, 4, 1, 5 };
		System.out.println(test.findPairs(pairs, 2));
		int[] pairs2 = { 1, 2, 3, 4, 5 };
		System.out.println(test.findPairs(pairs2, 1));

		System.out.println(test.complexNumberMultiply("1+1i", "1+1i"));
		System.out.println(test.complexNumberMultiply("1+-1i", "1+-1i"));

		int[][] o1matrix = { { 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 } };
		Arrays.stream(test.updateMatrix(o1matrix)).map(Arrays::toString).forEach(System.out::println);
		int[][] o1matrix2 = { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 1, 1 } };
		Arrays.stream(test.updateMatrix(o1matrix2)).map(Arrays::toString).forEach(System.out::println);

		int[] optimalDiv = { 2, 3, 4 };
		System.out.println(test.optimalDivision(optimalDiv));
		int[] optimalDiv2 = { 1000, 100, 10, 2 };
		System.out.println(test.optimalDivision(optimalDiv2));

		int[] subarray = { 1, 1, 1 };
		System.out.println(test.subarraySum(subarray, 2));
		int[] subarray2 = { 1, 2, 3 };
		System.out.println(test.subarraySum(subarray2, 3));
		int[] subarray3 = { -1, -1, 1 };
		System.out.println(test.subarraySum(subarray3, 0));

		TreeNode root = new TreeNode(3);
		root.left = new TreeNode(4);
		root.right = new TreeNode(5);
		root.left.left = new TreeNode(1);
		root.left.right = new TreeNode(2);
		TreeNode subroot = new TreeNode(4);
		subroot.left = new TreeNode(1);
		subroot.right = new TreeNode(2);
		System.out.println(test.isSubtree(root, subroot));

		int[] unsorted = { 2, 6, 4, 8, 10, 9, 15 };
		System.out.println(test.findUnsortedSubarray(unsorted));

		System.out.println(test.tree2str(root));

		System.out.println(test.solveEquation("x+5-3+x=6+x-2"));
		System.out.println(test.solveEquation("x=x"));
		System.out.println(test.solveEquation("2x=x"));

		int[] error = { 1, 2, 2, 4 };
		System.out.println(Arrays.toString(test.findErrorNums(error)));

		int[][] pairChain = { { 1, 2 }, { 2, 3 }, { 3, 4 } };
		System.out.println(test.findLongestChain(pairChain));
		int[][] pairChain2 = { { 1, 2 }, { 7, 8 }, { 4, 5 } };
		System.out.println(test.findLongestChain(pairChain2));

		System.out.println(
				test.cutOffTree(Arrays.asList(Arrays.asList(1, 2, 3), Arrays.asList(0, 0, 4), Arrays.asList(7, 6, 5))));
		System.out.println(
				test.cutOffTree(Arrays.asList(Arrays.asList(1, 2, 3), Arrays.asList(0, 0, 0), Arrays.asList(7, 6, 5))));
		System.out.println(
				test.cutOffTree(Arrays.asList(Arrays.asList(2, 3, 4), Arrays.asList(0, 0, 5), Arrays.asList(8, 7, 6))));

		String[] points = { "5", "2", "C", "D", "+" };
		System.out.println(test.calPoints(points));
		String[] points2 = { "5", "-2", "4", "C", "D", "9", "+", "+" };
		System.out.println(test.calPoints(points2));

		String[] wordFrequency = { "i", "love", "leetcode", "i", "love", "coding" };
		System.out.println(test.topKFrequent(wordFrequency, 2));
		String[] wordFrequency2 = { "the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is" };
		System.out.println(test.topKFrequent(wordFrequency2, 4));

		int[] pivot = { 1, 7, 3, 6, 5, 6 };
		System.out.println(test.pivotIndex(pivot));
		int[] pivot2 = { 2, 1, -1 };
		System.out.println(test.pivotIndex(pivot2));

		System.out.println(test.monotoneIncreasingDigits(10));
		System.out.println(test.monotoneIncreasingDigits(1234));
		System.out.println(test.monotoneIncreasingDigits(332));
		System.out.println(test.monotoneIncreasingDigits(0));

		System.out.println(test.countPrimeSetBits(6, 10));
		System.out.println(test.countPrimeSetBits(10, 15));

		System.out.println(test.partitionLabels("ababcbacadefegdehijhklij"));
		System.out.println(test.partitionLabels("eccbbbbdec"));

		System.out.println(test.reorganizeString("aab"));
		System.out.println(test.reorganizeString("aaab"));
		System.out.println(test.reorganizeString("aaabb"));

		int[] permutation = { 1, 0, 2 };
		System.out.println(test.isIdealPermutation(permutation));
		int[] permutation2 = { 1, 2, 0 };
		System.out.println(test.isIdealPermutation(permutation2));

		System.out.println(test.numUniqueEmails(new String[] { "test.email+alex@leetcode.com",
				"test.e.mail+bob.cathy@leetcode.com", "testemail+david@lee.tcode.com" }));
		System.out.println(test.numUniqueEmails(new String[] { "a@leetcode.com", "b@leetcode.com", "c@leetcode.com" }));

		System.out.println(test.knightDialer(1));
		System.out.println(test.knightDialer(3));
		System.out.println(test.knightDialer(2));
		System.out.println(test.knightDialer(3131));

		System.out.println(Arrays.toString(test.prisonAfterNDays(new int[] { 0, 1, 0, 1, 1, 0, 0, 1 }, 7)));
		System.out.println(Arrays.toString(test.prisonAfterNDays(new int[] { 1, 0, 0, 1, 0, 0, 1, 0 }, 1000000000)));

		int[][] kPoints = { { 1, 3 }, { -2, 2 } };
		Arrays.stream(test.kClosest(kPoints, 1)).map(Arrays::toString).forEach(System.out::println);
		int[][] kPoints2 = { { 3, 3 }, { 5, -1 }, { -2, 4 } };
		Arrays.stream(test.kClosest(kPoints2, 2)).map(Arrays::toString).forEach(System.out::println);

		System.out.println(test.subarraysWithKDistinct(new int[] { 1, 2, 1, 2, 3 }, 2));
		System.out.println(test.subarraysWithKDistinct(new int[] { 1, 2, 1, 3, 4 }, 3));

		int[][] oranges = { { 2, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } };
		System.out.println(test.orangesRotting(oranges));
		int[][] oranges2 = { { 2, 1, 1 }, { 1, 1, 0 }, { 0, 1, 1 } };
		System.out.println(test.orangesRotting(oranges2));
		int[][] oranges3 = { { 0, 2 } };
		System.out.println(test.orangesRotting(oranges3));

		List<List<Integer>> servers = Arrays.asList(Arrays.asList(0, 1), Arrays.asList(1, 2), Arrays.asList(2, 0),
				Arrays.asList(1, 3));
		System.out.println(test.criticalConnections(4, servers));
	}

	private static void printLinkedList(ListNode node) {
		if (node != null) {
			System.out.printf("%d ", node.val);
			printLinkedList(node.next);
		}
	}

}

class ListNode {
	int val;
	ListNode next;

	ListNode() {
	}

	ListNode(int val) {
		this.val = val;
		this.next = null;
	}

	ListNode(int val, ListNode next) {
		this.val = val;
		this.next = next;
	}
}

class TreeNode {
	int val;
	TreeNode left;
	TreeNode right;

	TreeNode() {
	}

	TreeNode(int val) {
		this.val = val;
	}

	TreeNode(int val, TreeNode left, TreeNode right) {
		this.val = val;
		this.left = left;
		this.right = right;
	}
}

class Node {
	int val;
	Node next;
	Node random;

	public Node(int val) {
		this.val = val;
		this.next = null;
		this.random = null;
	}
}

class TrieNode {

	TrieNode[] next = new TrieNode[26];
	String word;
	TrieNode[] children = new TrieNode[26];
	boolean isWord;
	TrieNode[] c;

	public TrieNode() {
		isWord = false;
		c = new TrieNode[128];
	}
}

class SearchNode {
	SearchNode[] next = new SearchNode[26];
	boolean end;
	String word;

	SearchNode() {
		for (int i = 0; i < 26; i++) {
			next[i] = null;
		}
		end = false;
		word = "";
	}

	int wordCount = 0;
}