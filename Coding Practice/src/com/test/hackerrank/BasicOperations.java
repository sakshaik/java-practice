package com.test.hackerrank;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

public class BasicOperations {

	public static void main(String[] args) {
		System.out.println(BasicOperations.slidingWindowOfSizeK(Arrays.asList(1, 2, 1, 4, 2, 1), 3, 2));
		System.out.println(BasicOperations.migratoryBirds(Arrays.asList(1, 4, 4, 4, 5, 3)));
		System.out.println(BasicOperations.migratoryBirds(Arrays.asList(1, 1, 2, 2, 3)));
		System.out.println(BasicOperations.isPallindrome(121));
		System.out.println(BasicOperations.isPallindrome(-121));
		System.out.println(Arrays.toString(BasicOperations.longestUniformSubstring("aabbb")));
		System.out.println(Arrays.toString(BasicOperations.longestUniformSubstring("AAac")));
		System.out.println(Arrays.toString(BasicOperations.longestUniformSubstring("aaaAAAAbbbcbcb")));
		System.out.println(BasicOperations.runLengthEncoding("aabbbBBcccCdeef"));
		System.out.println(BasicOperations.maxWaterCollected(new int[] { 0, 1, 3, 0, 1, 2, 0, 4, 2, 0, 3, 0 }));
		System.out.println(BasicOperations.isPangram("The quick brown fox jumps over the lazy dog"));
		System.out.println(BasicOperations.isPangram("aabbaaccczzz!!!!!"));
	}

	// Two Sets
	/*
	 * The elements of the first array are all factors of the integer being
	 * considered The integer being considered is a factor of all elements of the
	 * second array
	 */
	public static int getTotalX(List<Integer> a, List<Integer> b) {
		int lcm = a.get(0);
		for (int i = 1; i < a.size(); i++) {
			lcm = lcm(lcm, a.get(i));
		}
		int gcd = b.get(0);
		for (int i = 1; i < b.size(); i++) {
			gcd = gcd(gcd, b.get(i));
		}
		int count = 0;
		for (int i = lcm, j = 2; i <= gcd; i = lcm * j, j++) {
			if (gcd % i == 0) {
				count++;
			}
		}
		return count;
	}

	private static int lcm(int a, int b) {
		return a * (b / gcd(a, b));
	}

	public static int gcd(int a, int b) {
		while (b > 0) {
			int temp = b;
			b = a % b;
			a = temp;
		}
		return a;
	}

	public static void staircase(int n) {
		int l = 0, r = n - 1;
		while (l < n && r >= 0) {
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < r; i++) {
				sb.append(" ");
			}
			for (int j = r; j <= n - 1; j++) {
				sb.append("#");
			}
			l++;
			r--;
			System.out.println(sb.toString());
		}
	}

	// max value frequence in an array
	public static int birthdayCakeCandles(List<Integer> candles) {
		Map<Integer, Integer> m = new HashMap<Integer, Integer>();
		int max = Integer.MIN_VALUE;
		for (Integer candle : candles) {
			m.put(candle, m.getOrDefault(candle, 0) + 1);
			if (candle > max) {
				max = candle;
			}
		}
		return m.get(max);
	}

	public static String timeConversion(String s) {
		try {
			DateFormat df = new SimpleDateFormat("hh:mm:ssa");
			DateFormat out = new SimpleDateFormat("HH:mm:ss");
			Date date = df.parse(s);
			return out.format(date);
		} catch (ParseException e) {
			e.printStackTrace();
		}
		return null;
	}

	// Sliding window of side m - equal to sum d
	public static int birthday(List<Integer> s, int d, int m) {
		int count = 0;
		for (int i = 0; i <= s.size() - m; i++) {
			int sum = 0;
			for (int j = i; j < m + i; j++) {
				sum += s.get(j);
			}
			if (sum == d) {
				count++;
			}
		}
		return count;
	}

	public static int slidingWindowOfSizeK(List<Integer> s, int d, int m) {
		Queue<Integer> q = new LinkedList<Integer>();
		int i = 0;
		for (; i < m; i++) {
			q.add(s.get(i));
		}
		int count = 0;
		while (i <= s.size()) {
			if (q.stream().mapToInt(x -> x).sum() == d) {
				count++;
			}
			q.remove();
			if (i != s.size()) {
				q.add(s.get(i));
			}
			i++;
		}
		return count;
	}

	// Get all subarray of size k
	public static int migratoryBirds(List<Integer> arr) {
		int count = 0;
		Map<Integer, Integer> m = new HashMap<Integer, Integer>();
		for (int i = 0; i < arr.size(); i++) {
			m.put(arr.get(i), m.getOrDefault(arr.get(i), 0) + 1);
		}
		int value = Integer.MIN_VALUE;
		for (Integer key : m.keySet()) {
			if (m.get(key) > count) {
				count = m.get(key);
				if (value < key) {
					value = key;
				}
			}
		}
		return value;
	}

	public static String dayOfProgrammer(int year) {
		if (year == 1918)
			return "26.09.1918";
		if (isLeap(year))
			return "12.09." + Integer.toString(year);
		else
			return "13.09." + Integer.toString(year);
	}

	public static boolean isLeap(int year) {
		if (year % 4 != 0)
			return false;
		if (year > 1918 && year % 100 == 0 && year % 400 != 0)
			return false;
		return true;
	}

	public static boolean isPallindrome(int x) {
		int reverse = 0;
		int temp = x;
		while (temp > 0) {
			reverse = reverse * 10 + temp % 10;
			temp = temp / 10;
		}
		if (reverse == x) {
			return true;
		}
		return false;
	}

	public String longestCommonPrefix(String[] strs) {
		if (strs.length == 1) {
			return strs[0];
		}

		String first = strs[0];
		int checkCount = 0; // To check if the substring matches with all the other strings in arr
		String s = "";

		String smallest = strs[0]; // To get the size of the smallest string in the arr
		for (int i = 1; i < strs.length; i++) {
			if (strs[i].length() < smallest.length()) {
				smallest = strs[i];
			}
		}

		// Checking the substring based on the smallest size of string found above
		for (int i = smallest.length() - 1; i >= 0; i--) {
			String subs = first.substring(0, i + 1);
			s = subs;
			for (int j = 1; j < strs.length; j++) {
				if (strs[j].substring(0, i + 1).equals(subs)) {
					checkCount += 1;
				} else {
					break;
				}
			}
			if (checkCount == strs.length - 1) {
				break;
			}
			checkCount = 0;
		}
		if (checkCount == 0) {
			return "";
		}
		return s;
	}

	public int maxSubArray(int[] nums) {
		int cur = nums[0];
		int total = nums[0];
		for (int i = 1; i < nums.length; i++) {
			if (cur < 0) {
				cur = nums[i];
			} else {
				cur += nums[i];
			}
			if (cur > total) {
				total = cur;
			}
		}
		return total;
	}

	public int lengthOfLastWord(String s) {
		int n = s.length();
		int current = 0, lastWord = 0;
		for (int i = 0; i < n; i++) {
			if (s.charAt(i) == ' ') {
				if (current != 0)
					lastWord = current;
				current = 0;
				continue;
			}
			current++;
		}
		return current != 0 ? current : lastWord;
	}

	public int[] plusOne(int[] digits) {
		int rem = 1;
		for (int i = digits.length - 1; i >= 0; i--) {
			int tmp = digits[i] + rem;
			digits[i] = tmp % 10;
			rem = tmp / 10;
		}
		if (rem != 0) {
			int arr[] = new int[digits.length + 1];
			arr[0] = rem;
			for (int i = 1; i < arr.length; i++) {
				arr[i] = digits[i - 1];
			}
			return arr;
		}
		return digits;
	}

	public String addBinary(String a, String b) {
		StringBuilder sb = new StringBuilder();
		int i = a.length() - 1; // i is the pointer of "a" string
		int j = b.length() - 1; // j is the pointer of "b" string
		int carry = 0;
		while (i >= 0 || j >= 0) {
			int sum = carry;
			if (i >= 0)
				sum += a.charAt(i) - '0'; // to get the value like 1 - 0 = 1 & 0 - 0 = 0
			if (j >= 0)
				sum += b.charAt(j) - '0';

			sb.append(sum % 2);
			carry = sum / 2;

			i--;
			j--;

		}
		if (carry != 0)
			sb.append(carry);
		return sb.reverse().toString();
	}

	// Min steps to climb n stairs with 1 or 2 step at a time
	public int climbStairs(int n) {
		if (n == 1)
			return 1;
		int first = 1, second = 2, i = 3, temp;
		while (i <= n) {
			temp = second;
			second = first + second;
			first = temp;
			i += 1;
		}
		return second;
	}

	public void merge(int[] nums1, int m, int[] nums2, int n) {
		int i = m - 1;
		int j = n - 1;
		int index = m + n - 1;
		while (j >= 0) {
			if (i >= 0 && nums1[i] > nums2[j]) {
				nums1[index] = nums1[i];
				i--;
			} else {
				nums1[index] = nums2[j];
				j--;
			}
			index--;
		}
	}

	public static int[] longestUniformSubstring(String s) {
		int n = s.length();
		int longestStart = -1;
		int longestCount = 0;
		int start = 0, end = 0;
		for (int i = 0; i < s.length(); i++) {
			start = i;
			while (i < n - 1 && s.charAt(i) == s.charAt(i + 1)) {
				i++;
			}
			end = i;
			int newcount = end + 1 - start;
			if (longestCount < newcount) {
				longestCount = newcount;
				longestStart = start;
			}
		}
		return new int[] { longestStart, longestCount };
	}

	public static String runLengthEncoding(String s) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < s.length(); i++) {
			int count = 1;
			while (i < s.length() - 1 && s.charAt(i) == s.charAt(i + 1)) {
				count++;
				i++;
			}
			sb.append(s.charAt(i)).append(count);
		}
		return sb.toString();
	}

	public static int maxWaterCollected(int[] count) {
		int result = 0;
		int n = count.length;
		int[] left = new int[n];
		left[0] = count[0];
		for (int i = 1; i < n; i++) {
			left[i] = Math.max(count[i], left[i - 1]);
		}
		int[] right = new int[n];
		right[n - 1] = count[n - 1];
		for (int j = n - 2; j >= 0; j--) {
			right[j] = Math.max(count[j], right[j + 1]);
		}
		for (int k = 0; k < n; k++) {
			result += Math.min(left[k], right[k]) - count[k];
		}
		return result;
	}

	public static String isPangram(String s) {
		StringBuffer sb = new StringBuffer();
		Map<Character, Integer> map = new HashMap<Character, Integer>();
		for (int i = 0; i < 256; i++) {
			if ((i >= 'a' && i <= 'z') || (i >= 'A' && i <= 'Z')) {
				map.put((char) i, 0);
			}
		}
		for (int i = 0; i < s.length(); i++) {
			if ((s.charAt(i) >= 'a' && s.charAt(i) <= 'z') || (s.charAt(i) >= 'A' && s.charAt(i) <= 'Z')) {
				map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
			}
		}
		for (Character key : map.keySet()) {
			if (map.get(key) == 0) {
				sb.append(key);
			}
		}
		return sb.toString();
	}

}
