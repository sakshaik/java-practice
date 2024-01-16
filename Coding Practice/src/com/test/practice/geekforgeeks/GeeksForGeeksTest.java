package com.test.practice.geekforgeeks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;

public class GeeksForGeeksTest {

	public static void main(String[] args) {
		GeeksForGeeksTest test = new GeeksForGeeksTest();

		// Alternating maximum
		System.out.println(test.maxPizza(5, new int[] { 1, 2, 3, 4, 5 }));

		// Find if substring is an anagram
		System.out.println(test.valid("zyxabc", "fghcxabyzbvf"));
		System.out.println(test.valid("htc", "hxtcczht"));

		// Repeat sibsequence of fraction
		System.out.println(test.findRepeat(1, 2));
		System.out.println(test.findRepeat(22, 7));
		System.out.println(test.findRepeat(8, 4));

		// Maximum sum product ot two unequal arrays
		System.out.println(test.project(new long[] { 4, 1, 5, 2, 9 }, new long[] { 1, 2, 1, 1 }, 5));
		System.out.println(test.project(new long[] { 4, 1 }, new long[] { 1 }, 2));
		System.out.println(test.project(new long[] { 42, 0, 14, 26, 37, 16, 1, 13, 35, 3, 40 },
				new long[] { 31, 1, 44, 32, 20, 27, 40, 6, 6, 16 }, 11));
		System.out.println(test.project(new long[] { 42, 22, 36, 40, 9, 47 }, new long[] { 40, 41, 47, 43, 15 }, 6));
		System.out.println(test.project2(new long[] { 4, 1, 5, 2, 9 }, new long[] { 1, 2, 1, 1 }, 5));
		System.out.println(test.project2(new long[] { 4, 1 }, new long[] { 1 }, 2));
		System.out.println(test.project2(new long[] { 42, 0, 14, 26, 37, 16, 1, 13, 35, 3, 40 },
				new long[] { 31, 1, 44, 32, 20, 27, 40, 6, 6, 16 }, 11));
		System.out.println(test.project2(new long[] { 42, 22, 36, 40, 9, 47 }, new long[] { 40, 41, 47, 43, 15 }, 6));

		// No of Triplets equal to 0
		System.out.println(GeeksForGeeksTest.ways(new int[] { 9, 13, 14, -16, 9, 12, 15, 0, -1, 13, 14, 6 }, 12));
		System.out.println(GeeksForGeeksTest.ways(new int[] { -1, 0, 1, 1, 2, -1, -4 }, 7));
		System.out.println(GeeksForGeeksTest.ways(new int[] { -1, 1, 2, 4, 1, -2, -3 }, 7));

		// Cars in an accident
		System.out.println(test.finalSeq(new int[] { 100, 10, -2 }, 3));
		System.out.println(test.finalSeq(new int[] { 3, -9, 8, -8, 9 }, 5));

		// Difference of characters in two strings
		System.out.println(test.need_to_carry("abcdab", "abad"));
		System.out.println(test.need_to_carry("aghahb", "ahagb"));

		// Factor combinations
		System.out.println(test.getFactors(12));
		System.out.println(test.getFactors(8));

		// Path from Top left to bottom right
		int[][] golf = { { 3, 0, 0, 2 }, { 0, 2, 1, 1 }, { 1, 0, 0, 0 }, { 0, 4, 0, 0 } };
		System.out.println(test.is_possible(golf, 4));

		// Counting Letters and replace
		test.Count_V_W(new StringBuilder("ababb"), 5, 7);
		test.Count_V_W(new StringBuilder("aabab"), 5, 7);

		int[][] food = { { 10, 4 }, { 3, 5 } };
		System.out.println(test.strength(food, 2, 2));

		// Maximum in sub array of size k
		GeeksForGeeksTest.parcel_store(new int[] { 10, 5, 1, 2, 7 }, 5, 2);

	}

	int max = 0;

	int longest_strictly_increasing_path(Node head) {
		max = 0;
		travel(head, 1);
		return max;
	}

	int travel(Node n, int change) {
		int inc_max = 0;
		int dec_max = 0;

		if (n.left != null) {
			if (n.left.data > n.data)
				inc_max = travel(n.left, 1);
			else if (n.left.data < n.data)
				dec_max = travel(n.left, 0);
			else
				travel(n.left, 1);
		}
		if (n.right != null) {
			int temp1 = 0, temp2 = 0;
			if (n.right.data > n.data)
				temp1 = travel(n.right, 1);
			else if (n.right.data < n.data)
				temp2 = travel(n.right, 0);
			else
				travel(n.right, 1);
			if (temp1 > inc_max)
				inc_max = temp1;
			if (temp2 > dec_max)
				dec_max = temp2;
		}
		if (inc_max + dec_max + 1 > max)
			max = inc_max + dec_max + 1;

		if (change == 1)
			return inc_max + 1;
		return dec_max + 1;
	}

	// Alternate max sum
	int maxPizza(int n, int[] arr) {
		int incl = arr[0];
		int excl = 0;
		int excl_new;

		for (int i = 1; i < n; i++) {
			excl_new = (incl > excl) ? incl : excl;
			incl = excl + arr[i];
			excl = excl_new;
		}
		return ((incl > excl) ? incl : excl);
	}

	// Is all leaf nodes at a difference of one
	boolean isPossible(Node root) {
		if (root == null) {
			return true;
		}
		Queue<Node> q = new LinkedList<Node>();
		q.add(root);
		int check = Integer.MAX_VALUE;
		int level = 0;
		while (!q.isEmpty()) {
			level++;
			int size = q.size();
			while (size > 0) {
				Node temp = q.remove();
				if (temp.left != null) {
					q.add(temp.left);
					if (temp.left.left == null && temp.left.right == null) {
						if (check == Integer.MAX_VALUE) {
							check = level;
						} else if (Math.abs(check - level) > 1) {
							return false;
						}
					}
				}

				if (temp.right != null) {
					q.add(temp.right);
					if (temp.right.left == null && temp.right.right == null) {
						if (check == Integer.MAX_VALUE) {
							check = level;
						} else if (Math.abs(check - level) > 1) {
							return false;
						}
					}
				}

				size--;
			}
		}
		return true;
	}

	// Is password an anagram of the given string
	boolean valid(String p, String s) {
		int s2hash[] = new int[26];
		int s1hash[] = new int[26];
		int s1len = p.length();
		int s2len = s.length();

		if (s1len > s2len)
			return false;
		int left = 0, right = 0;
		while (right < s1len) {
			s1hash[p.charAt(right) - 'a'] += 1;
			s2hash[s.charAt(right) - 'a'] += 1;
			right++;
		}
		right -= 1;
		while (right < s2len) {
			if (Arrays.equals(s1hash, s2hash))
				return true;
			right++;
			if (right != s2len)
				s2hash[s.charAt(right) - 'a'] += 1;

			s2hash[s.charAt(left) - 'a'] -= 1;
			left++;
		}
		return false;
	}

	// Repeated fraction length
	int findRepeat(int p, int q) {
		String res = "";
		HashMap<Integer, Integer> mp = new HashMap<>();
		int rem = p % q;
		while ((rem != 0) && (!mp.containsKey(rem))) {
			mp.put(rem, res.length());
			rem = rem * 10;
			int res_part = rem / q;
			res += String.valueOf(res_part);
			rem = rem % q;
		}
		if (rem == 0)
			return 0;
		else if (mp.containsKey(rem))
			return res.substring(mp.get(rem)).length();
		return 0;
	}

	// Exclude one algo expert and maximize output
	long project(long algoExperts[], long Developers[], int n) {
		Queue<Long> q = new LinkedList<Long>();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i != j) {
					q.add(algoExperts[j]);
				}
			}
		}
		long result = 0;
		while (!q.isEmpty()) {
			long sum = 0;
			for (int j = 0; j < n - 1; j++) {
				sum += Developers[j] * q.poll();
			}
			result = Math.max(result, sum);
		}
		return result;
	}

	long project2(long algoExperts[], long Developers[], int n) {
		long dpf[] = new long[n + 5];
		long dpb[] = new long[n + 5];
		long a[] = new long[n + 5];
		long b[] = new long[n + 5];

		long ans = 0;

		// Making the array 1 indexed.
		for (int i = 0; i < n; i++)
			a[i + 1] = algoExperts[i];
		for (int i = 0; i < n - 1; i++)
			b[i + 1] = Developers[i];

		// Calculating the product sum from front
		for (int i = 1; i <= n; i++)
			dpf[i] = dpf[i - 1] + a[i] * b[i];

		// Calculating the product sum from back
		for (int i = n; i > 0; i--)
			dpb[i] = dpb[i + 1] + a[i] * b[i - 1];

		// placing zero at (i+1)th index and storing the maximum answer
		for (int i = 0; i < n; i++)
			ans = max(dpf[i] + dpb[i + 2], ans);

		return ans;
	}

	long max(long a, long b) {
		if (a > b)
			return a;
		else
			return b;
	}

	static int ways(int a[], int n) {
		HashSet<String> s = new HashSet<String>();
		Arrays.sort(a);
		for (int i = 0; i < n - 2; i++) {
			if (a[i] > 0) {
				break;
			}
			int l = i + 1;
			int r = n - 1;
			while (l < r) {
				int total = a[i] + a[l] + a[r];
				if (total < 0) {
					l = l + 1;
				} else if (total > 0) {
					r = r - 1;
				} else {
					String s1 = Integer.toString(a[i]) + "," + Integer.toString(a[l]) + "," + Integer.toString(a[r]);
					s.add(s1);
					l = l + 1;
					r = r - 1;
				}
			}
		}
		return s.size();
	}

	public ArrayList<Integer> finalSeq(int cars[], int n) {
		Stack<Integer> s = new Stack<Integer>();
		int flag;
		for (int i = 0; i < n; i++) {
			flag = 0;
			while (!s.empty() && cars[i] < 0 && s.peek() > 0) {
				if (s.peek() < -cars[i]) {
					s.pop();
					continue;
				} else if (s.peek() == -cars[i])
					s.pop();
				flag = 1;
				break;
			}
			if (flag == 0)
				s.push(cars[i]);
		}

		ArrayList<Integer> ans = new ArrayList<Integer>();
		while (!s.empty()) {
			ans.add(s.peek());
			s.pop();
		}
		Collections.reverse(ans);
		return ans;
	}

	public int need_to_carry(String s1, String s2) {
		int cArr[] = new int[26];
		for (int i = 0; i < s1.length(); i++)
			cArr[s1.charAt(i) - 97]++;
		for (int i = 0; i < s2.length(); i++)
			cArr[s2.charAt(i) - 97]--;
		int count = 0;
		for (int i = 0; i < 26; i++)
			count += Math.abs(cArr[i]);
		return count;
	}

	int getFactors(int n) {
		List<List<Integer>> result = new LinkedList<List<Integer>>();
		process(2, result, new LinkedList<Integer>(), n);
		return result.size();
	}

	private void process(int start, List<List<Integer>> result, List<Integer> curr, int n) {
		if (n == 1) {
			if (curr.size() > 1) {
				result.add(new LinkedList<Integer>(curr));
			}
		}
		for (int i = start; i <= n; i++) {
			if (n % i == 0) {
				curr.add(i);
				process(i, result, curr, n / i);
				curr.remove(curr.size() - 1);
			}
		}
	}

	int is_possible(int a[][], int n) {
		return dfs(a, 0, 0, n);
	}

	int dfs(int a[][], int r, int c, int n) {
		if (r == n - 1 && c == n - 1)
			return 1;
		if (a[r][c] == 0)
			return 0;
		int m = a[r][c];
		a[r][c] = 0;
		for (int i = m; i >= 0; i = i - 1) {
			if (r + i < n) {
				if (dfs(a, r + i, c, n) == 1)
					return 1;
			}
			if (c + i < n) {
				if (dfs(a, r, c + i, n) == 1)
					return 1;
			}
			if (r - i >= 0) {
				if (dfs(a, r - i, c, n) == 1)
					return 1;
			}
			if (c - i >= 0) {
				if (dfs(a, r, c - i, n) == 1)
					return 1;
			}
		}
		return 0;
	}

	void Count_V_W(StringBuilder s, int n, int k) {
		// variable for counting the consonant and vowel
		int con = 0, vow = 0;
		// counting number of vowels before the particular index
		int till = 0;
		// counting vowels and consonant in the given string
		for (int i = 0; i < n; i++) {
			if (check(s.charAt(i)))
				vow++;
			else
				con++;
		}
		for (int i = 0; i < k; i++) {
			// if vowel
			if (check(s.charAt(i))) {
				// if size of current string is less than k then append "ma"
				if (s.length() < k) {
					s.append("ma");
					// logic is basically we have to create the size of s just
					// greater than k.
				}
				till++;
				con++;
			} else {
				// Here also as per the rule of the inserting we have to insert
				// the K in S.
				if (s.length() < k)
					s.append("m");
				int i1 = 0;
				// push the char 'a' as many times as it arrived previously.
				while (i1 < till && s.length() < k) {
					s.append("a");
					i1++;
				}
				vow += till;
			}
		}
		System.out.println(vow + " " + con);
	}

	boolean check(char a) {
		if (a == 'a' || a == 'e' || a == 'i' || a == 'o' || a == 'u' || a == 'A' || a == 'E' || a == 'I' || a == 'O'
				|| a == 'U')
			return true;
		else
			return false;
	}

	public int strength(int[][] grid, int n, int m) {
		int i, j, maxx = Integer.MIN_VALUE;
		int[][] dp = new int[n][m];
		for (i = 0; i < m; i++)
			dp[n - 1][i] = grid[n - 1][i];
		for (i = n - 2; i >= 0; i--) {
			for (j = 0; j < m; j++) {
				if (j > 0 && j < m - 1)
					dp[i][j] = Math.max(grid[i][j] + dp[i + 1][j + 1],
							Math.max(grid[i][j] + dp[i + 1][j], grid[i][j] + dp[i + 1][j - 1]));
				else if (j == 0 && j < m - 1)
					dp[i][j] = Math.max(grid[i][j] + dp[i + 1][j + 1], grid[i][j] + dp[i + 1][j]);
				else if (j == m - 1 && j > 0)
					dp[i][j] = Math.max(grid[i][j] + dp[i + 1][j], grid[i][j] + dp[i + 1][j - 1]);
				else
					dp[i][j] = grid[i][j] + dp[i + 1][j];
			}
		}
		for (i = 0; i < m; i++) {
			if (dp[0][i] > maxx)
				maxx = dp[0][i];
		}
		if (maxx < 0)
			return 0;
		return maxx;
	}

	ArrayList<Integer> leaves = new ArrayList<Integer>();
	ArrayList<Integer> minimum = new ArrayList<Integer>();

	public int maxRootToLeafPathSum(Node root) {
		leaves.clear();
		minimum.clear();
		func(root, 0, Integer.MAX_VALUE);
		int ans = Integer.MIN_VALUE;
		Iterator<Integer> itr = leaves.iterator();
		Iterator<Integer> mtr = minimum.iterator();
		while (itr.hasNext()) {
			int a, b;
			a = itr.next().intValue();
			b = mtr.next().intValue();
			ans = Math.max(ans, Math.max(a, a - 2 * b));
		}
		return ans;
	}

	public void func(Node root, int sum, int mini) {
		if (root == null)
			return;
		if (root.left == null && root.right == null) {
			leaves.add(sum + root.data);
			minimum.add(Math.min(mini, root.data));
			return;
		}
		func(root.left, sum + root.data, Math.min(mini, root.data));
		func(root.right, sum + root.data, Math.min(mini, root.data));
	}

	static void parcel_store(int arr[], int n, int k) {
		Deque<Integer> Qi = new LinkedList<Integer>();
		int i;
		for (i = 0; i < k; ++i) {
			while (!Qi.isEmpty() && arr[i] >= arr[Qi.peekLast()])
				Qi.removeLast();
			Qi.addLast(i);
		}

		for (; i < n; ++i) {
			System.out.print(arr[Qi.peek()] + " ");
			while ((!Qi.isEmpty()) && Qi.peek() <= i - k)
				Qi.removeFirst();
			while ((!Qi.isEmpty()) && arr[i] >= arr[Qi.peekLast()])
				Qi.removeLast();
			Qi.addLast(i);
		}
		System.out.print(arr[Qi.peek()]);
	}

	public int min_cost(int[] arr, int n) {
		int i;
		int[] dp = new int[n];
		dp[0] = arr[0];
		dp[3] = dp[0] + arr[3];
		dp[1] = dp[3] + arr[1] + arr[2];
		dp[2] = dp[3] + arr[2];

		for (i = 4; i <= n - 3; i++) {
			dp[i] = arr[i] + Math.min(dp[i - 1] + arr[i + 1] + arr[i + 2], Math.min(dp[i - 3], dp[i - 2] + arr[i + 1]));
		}

		dp[n - 1] = dp[n - 4] + arr[n - 1];
		return dp[n - 1];
	}

	public String lexicoSmallestSubSeq(String s, int k) {
		ArrayList<LinkedList<Integer>> letters = new ArrayList<LinkedList<Integer>>();
		for (int i = 0; i < 26; i++) {
			letters.add(new LinkedList<Integer>());
		}
		for (int i = 0; i < s.length(); i++) {
			letters.get(s.charAt(i) - 'a').add(i);
		}
		StringBuilder sb = new StringBuilder();
		int lowerLimit = 0, upperLimit = 0;
		while (k-- > 0) {
			upperLimit = s.length() - k - 1;
			int pick = 0;
			while (true) {
				while (!letters.get(pick).isEmpty() && lowerLimit > letters.get(pick).peek()) {
					letters.get(pick).remove();
				}
				if (!letters.get(pick).isEmpty() && upperLimit >= letters.get(pick).peek()) {
					break;
				}
				pick++;
			}
			sb.append((char) (pick + 'a'));
			lowerLimit = letters.get(pick).peek() + 1;
		}
		return sb.toString();
	}

	public int countNodes(Node root) {
		int depthoftree = depth(root) - 1;
		// Condition for only root present
		if (depthoftree == 0)
			return 1;

		// Lower bound of no of nodes
		int l = 1;
		// Upper bound of no of nodes
		int r = (int) Math.pow(2, depthoftree) - 1;

		while (l <= r) {
			int mid = (l + r) / 2;

			if (midExists(root, mid, depthoftree)) {
				l = mid + 1;
			} else
				r = mid - 1;
		}

		return (int) Math.pow(2, depthoftree) - 1 + l;
	}
	
	public int depth(Node root) {
		if (root == null)
			return 0;
		// Recur only to left as
		// it is a complete binary tree
		return 1 + depth(root.left);
	}

	// Function to check if the node mid exists in the tree or not
	public boolean midExists(Node root, int mid, int depthoftree) {

		// Lower bound of no of nodes
		int l = 0;
		// Upper bound of no of nodes
		int r = (int) Math.pow(2, depthoftree) - 1;

		// Search for the existence of node mid
		for (int i = 0; i < depthoftree; i++) {
			// m is leftmost of current root
			int m = (l + r) / 2;

			// check if mid is present
			// in left subtree
			if (mid <= m) {
				// Search in left subtree
				root = root.left;
				r = m;
			} else {
				// Search in right subtree
				root = root.right;
				l = m;
			}
		}

		if (root == null)
			return false;

		return true;
	}

	
}

class Node {
	int data;
	Node left;
	Node right;

	Node(int data) {
		this.data = data;
		left = null;
		right = null;
	}
}