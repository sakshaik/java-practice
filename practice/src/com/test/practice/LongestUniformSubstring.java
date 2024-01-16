package com.test.practice;

import java.util.Arrays;

public class LongestUniformSubstring {

	public static void main(String[] args) {
		System.out.println(Arrays.toString(LongestUniformSubstring.longestUniformSubstring("aaabb")));
		System.out.println(Arrays.toString(LongestUniformSubstring.longestUniformSubstring("AAac")));
		System.out.println(Arrays.toString(LongestUniformSubstring.longestUniformSubstring("aabbb")));
	}

	private static int[] longestUniformSubstring(String s) {
		int longestCount = 0, longestStart = -1, n = s.length(), start = 0, end = 0;
		for (int i = 0; i < n; i++) {
			start = i;
			while (i < n - 1 && s.charAt(i) == s.charAt(i + 1)) {
				i++;
			}
			end = i;
			int newCount = end + 1 - start;
			if (longestCount < newCount) {
				longestCount = newCount;
				longestStart = start;
			}
		}
		return new int[] { longestStart, longestCount };
	}

}
