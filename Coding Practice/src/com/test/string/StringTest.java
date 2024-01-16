package com.test.string;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class StringTest {

	public static void main(String[] args) {
		System.out.println(StringTest.mostCommonWord("I will not tell who to tell"));
		System.out.println(StringTest.lengthOfLastWord("I will not tell who to tell"));
		System.out.println(StringTest.repeatedSubstringPattern("abab"));
		System.out.println(StringTest.repeatedSubstringPattern("ababab"));
		System.out.println(StringTest.repeatedSubstringPattern("ababbb"));
		System.out.println(Arrays.toString(StringTest.longestUniformSubstring("aabbb")));
		System.out.println(Arrays.toString(StringTest.longestUniformSubstring("AAac")));
		System.out.println(Arrays.toString(StringTest.longestUniformSubstring("aaaAAAAbbbcbcb")));
		System.out.println(StringTest.runLengthEncoding("aabbbBBcccCdeef"));
		System.out.println(StringTest.runLengthEncoding("abbcccddddeeeeeffffff"));
		System.out.println(StringTest.longestCommonPrefix(new String[] { "abcd", "abcf", "abce" }));
	}

	private static String mostCommonWord(String paragraph) {
		String[] words = paragraph.split("[^a-zA-Z]+");
		Map<String, Integer> values = new HashMap<String, Integer>();
		for (int i = 0; i < words.length; i++) {
			words[i] = words[i].toLowerCase();
		}
		for (int i = 0; i < words.length; i++) {
			if (values.containsKey(words[i])) {
				int count = values.get(words[i]) + 1;
				values.put(words[i], count);
			} else {
				values.put(words[i], 1);
			}
		}
		int max = Integer.MIN_VALUE;
		String mostCommon = null;
		for (int i = 0; i < words.length; i++) {
			if (values.containsKey(words[i]) && values.get(words[i]) > max) {
				max = values.get(words[i]);
				mostCommon = words[i];
			}
		}
		return mostCommon;
	}

	public static String longestCommonPrefix(String[] strs) {
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

	public static boolean repeatedSubstringPattern(String s) {
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

	public static int lengthOfLastWord(String s) {
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

}
