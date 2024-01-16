package com.test.hackerrank;

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
import java.util.Set;
import java.util.Stack;
import java.util.TreeSet;

public class HackerRankTest {

	public static void main(String[] args) {

		// Sock Merchant
		System.out.println("Sock Merchant");
		List<Integer> ar = Arrays.asList(1, 2, 1, 2, 1, 3, 2);
		System.out.println(HackerRankTest.sockMerchant(ar.size(), ar));

		// Valley Test
		System.out.println("Valley Test");
		String path = "UDDDUDUU";
		System.out.println(HackerRankTest.countingValleys(path.length(), path));

		// Jumping in the clouds
		System.out.println("Jumping in the clouds");
		List<Integer> cloud = Arrays.asList(0, 1, 0, 0, 0, 1, 0);
		System.out.println(HackerRankTest.jumpingOnClouds(cloud));

		// Repeated String
		System.out.println("Repeated String");
		System.out.println(HackerRankTest.repeatedString("abcac", 10));
		System.out.println(HackerRankTest.repeatedString("dcd", 10));
		System.out.println(HackerRankTest.repeatedString("a", 10));

		// Hourglass sum
		System.out.println("Hourglass sum");
		int[][] array = { { -9, -9, -9, 1, 1, 1 }, { 0, -9, 0, 4, 3, 2 }, { -9, -9, -9, 1, 2, 3 }, { 0, 0, 8, 6, 6, 0 },
				{ 0, 0, 0, -2, 0, 0 }, { 0, 0, 1, 2, 4, 0 } };
		List<List<Integer>> list = new ArrayList<List<Integer>>();
		for (int i = 0; i < array.length; i++) {
			List<Integer> data = new ArrayList<Integer>();
			for (int j = 0; j < array[0].length; j++) {
				data.add(array[i][j]);
			}
			list.add(data);
		}
		System.out.println(list);
		System.out.println(HackerRankTest.hourglassSum(list));

		// Left Rotate array
		System.out.println("Left Rotate Array");
		List<Integer> nums = Arrays.asList(1, 3, 2, 4, 5, 6, 7);
		System.out.println(nums);
		System.out.println(HackerRankTest.rotLeft(nums, 3));

		// Mininum Bribes
		System.out.println("Minimum Bribes");
		List<Integer> position = new ArrayList<Integer>();
		int[] pos = { 1, 2, 5, 3, 7, 8, 6, 4 };
		for (int i = 0; i < pos.length; i++) {
			position.add(pos[i]);
		}
		HackerRankTest.minimumBribes(position);

		// Minimum swaps
		System.out.println("Minimum Swaps");
		System.out.println(HackerRankTest.minimumSwaps(pos));

		// Array manipulation
		List<Integer> query1 = Arrays.asList(1, 2, 100);
		List<Integer> query2 = Arrays.asList(2, 5, 100);
		List<Integer> query3 = Arrays.asList(3, 4, 100);
		List<List<Integer>> queries = new ArrayList<List<Integer>>();
		queries.add(query1);
		queries.add(query2);
		queries.add(query3);
		System.out.println("Array manipulation");
		System.out.println(HackerRankTest.arrayManipulation(5, queries));

		// Hashtable : Ransom Note
		System.out.println("Hashtable : Ransom Note");
		List<String> magazine = Arrays.asList("give", "me", "one", "grand", "today", "night");
		List<String> note = Arrays.asList("give", "one", "grand", "today");
		HackerRankTest.checkMagazine(magazine, note);
		magazine = Arrays.asList("two", "times", "three", "is", "not", "four");
		note = Arrays.asList("two", "times", "two", "is", "four");
		HackerRankTest.checkMagazine(magazine, note);

		// Two Strings have common substring
		System.out.println("Two String have common substring");
		System.out.println(HackerRankTest.twoStrings("abc", "ab"));
		System.out.println(HackerRankTest.twoStrings("abc", "def"));

		// Anagram list of a string
		System.out.println("Anagram list of a string");
		System.out.println(HackerRankTest.sherlockAndAnagrams("abba"));

		// Count triplets of geometric progression
		System.out.println("Count triplets of geometric progression");
		List<Long> values = Arrays.asList(1L, 3L, 9L, 9L, 27L, 81L);
		System.out.println(HackerRankTest.countTriplets(values, 3));

		// Frequency query

		// Bubble sort
		System.out.println("Bubble Sort");
		List<Integer> unsorted = Arrays.asList(3, 2, 1);
		HackerRankTest.countSwaps(unsorted);

		// Activity notification
		System.out.println("Activity notification");
		List<Integer> txns = Arrays.asList(10, 20, 30, 40, 50);
		System.out.println(HackerRankTest.activityNotifications(txns, 3));

		// Array Sort Swap Count
		System.out.println("Array Sort Swap Count");
		List<Integer> counts = Arrays.asList(2, 1, 3, 1, 2);
		System.out.println(HackerRankTest.countInversions(counts));
		System.out.println(counts);

		// Remove characters to make anagram
		System.out.println("Remove characters to make anagram");
		System.out.println(HackerRankTest.makeAnagram("fcrxzwscanmligyxyvym", "jxwtrhvujlmrpdoqbisbwhmgpmeoke"));

		// Alternating characters
		System.out.println("Remove characters to make string alternating");
		System.out.println(HackerRankTest.alternatingCharacters("AAABBB"));

		// Sherlock and the valid string
		System.out.println("Sherlock and the valid string");
		System.out.println(HackerRankTest.isValid("aabbccddeefghi"));
		System.out.println(HackerRankTest.isValid("abcc"));
		System.out.println(HackerRankTest.isValid("abccc"));

		// Special Substring count
		System.out.println("Special Substring count");
		String s = "mnonopoo";
		System.out.println(HackerRankTest.substrCount(s.length(), s));
		s = "abcbaba";
		System.out.println(HackerRankTest.substrCount(s.length(), s));

		// Common Child Algorithm
		System.out.println("Common Child Algorithm");
		System.out.println(HackerRankTest.commonChild("ABCD", "ABDC"));
		System.out.println(HackerRankTest.commonChild("SHINCHAN", "NOHARAAA"));
		System.out.println(HackerRankTest.commonChild("ABCDEF", "FBDAMN"));
		System.out.println(HackerRankTest.commonChild("AA", "BB"));

		// Minimum absolute difference
		System.out.println("Minimum absolute difference");
		List<Integer> min = Arrays.asList(3, -7, 0);
		System.out.println(HackerRankTest.minimumAbsoluteDifference(min));
		min = Arrays.asList(-59, -36, -13, 1, -53, -92, -2, -96, -54, 75);
		System.out.println(HackerRankTest.minimumAbsoluteDifference(min));

		// Luck Balance
		System.out.println("Luck Balance");
		List<List<Integer>> contests = Arrays.asList(Arrays.asList(5, 1), Arrays.asList(1, 1), Arrays.asList(4, 0));
		System.out.println(HackerRankTest.luckBalance(2, contests));
		System.out.println(HackerRankTest.luckBalance(1, contests));

		// Minimum cost for greedy florist
		System.out.println("Minimum cost for greedy florist");
		int[] flowers = { 1, 3, 5, 7, 9 };
		System.out.println(HackerRankTest.getMinimumCost(3, flowers));
		int[] flowers2 = { 2, 5, 6 };
		System.out.println(HackerRankTest.getMinimumCost(3, flowers2));
		System.out.println(HackerRankTest.getMinimumCost(2, flowers2));

		// Mininum Max-Min difference of a subarray
		System.out.println("Mininum Max-Min difference of a subarray");
		List<Integer> minMax = Arrays.asList(1, 4, 7, 2);
		System.out.println(HackerRankTest.maxMin(2, minMax));
		minMax = Arrays.asList(10, 100, 300, 200, 1000, 20, 30);
		System.out.println(HackerRankTest.maxMin(3, minMax));
		minMax = Arrays.asList(1, 2, 3, 4, 10, 20, 30, 40, 100, 200);
		System.out.println(HackerRankTest.maxMin(4, minMax));
		minMax = Arrays.asList(1, 2, 1, 2, 1);
		System.out.println(HackerRankTest.maxMin(2, minMax));

		// Reverse Shuffle Merge
		System.out.println("String - Reverse Shuffle Merge");
		System.out.println(HackerRankTest.reverseShuffleMerge("aeiouuoiea"));
		System.out.println(HackerRankTest.reverseShuffleMerge("abcdefgabcdefg"));
		System.out.println(HackerRankTest.reverseShuffleMerge("eggegg"));

		// What 2 flavors for a given value
		System.out.println("What 2 flavors for a given value");
		List<Integer> flavors = Arrays.asList(2, 1, 3, 5, 6);
		HackerRankTest.whatFlavors(flavors, 5);

		// Minimum pass to create candies
		System.out.println("Minimum pass to create candies");
		System.out.println(HackerRankTest.minimumPasses(1L, 2L, 1L, 60L));

		// Max Subarray Sum
		System.out.println("Max Subarray Sum");
		int[] a = { -2, 1, 3, -4, 5 };
		System.out.println(HackerRankTest.maxSubsetSum(a));

		// Abbreviation of two strings
		System.out.println("Abbreviation of two strings");
		System.out.println(HackerRankTest.abbreviation("AbcDE", "ABDE"));
		System.out.println(HackerRankTest.abbreviation("daBcd", "ABC"));
		System.out.println(HackerRankTest.abbreviation("AbcDE", "AFDE"));

		// Minimum candies by rating
		List<Integer> scores = Arrays.asList(4, 6, 4, 5, 6, 2);
		System.out.println(HackerRankTest.candies(scores.size(), scores));
		scores = Arrays.asList(1, 2, 2);
		System.out.println(HackerRankTest.candies(scores.size(), scores));

		// Largest rectangle plot for a mall
		System.out.println("Largest rectangle plot for a mall");
		List<Integer> heights = Arrays.asList(3, 2, 3);
		System.out.println(HackerRankTest.largestRectangle(heights));
		heights = Arrays.asList(1, 2, 3, 4, 5);
		System.out.println(HackerRankTest.largestRectangle(heights));
	}

	public static int sockMerchant(int n, List<Integer> ar) {
		HashMap<Integer, Integer> count = new HashMap<>();
		for (Integer color : ar) {
			count.put(color, count.getOrDefault(color, 0) + 1);
		}
		int result = 0;
		for (Integer color : count.keySet()) {
			result += count.get(color) / 2;
		}
		return result;
	}

	public static int countingValleys(int steps, String path) {
		int v = 0;
		int lvl = 0;
		for (char c : path.toCharArray()) {
			if (c == 'U')
				++lvl;
			if (c == 'D')
				--lvl;

			if (lvl == 0 && c == 'U')
				++v;
		}
		return v;
	}

	public static int jumpingOnClouds(List<Integer> c) {
		int count = -1;
		for (int i = 0; i < c.size(); i++, count++) {
			if (i < c.size() - 2 && c.get(i + 2) == 0)
				i++;
		}
		return count;
	}

	public static long repeatedString(String s, long n) {
		char[] c = s.toLowerCase().toCharArray();
		int l = c.length;
		if (l == 1 && c[0] == 'a') {
			return n;
		}
		int count = 0;
		for (int i = 0; i < l; i++) {
			if (c[i] == 'a') {
				count++;
			}
		}
		if (count == 0) {
			return count;
		}
		if (n % l == 0) {
			return (n / l) * count;
		} else {
			long repetition = (n / l) * count;
			long remainder = n % l;
			int temp = 0;
			for (int i = 0; i < remainder; i++) {
				if (c[i] == 'a') {
					temp++;
				}
			}
			return repetition + temp;
		}
	}

	public static int hourglassSum(List<List<Integer>> arr) {
		int result = Integer.MIN_VALUE;
		int rows = arr.size();
		int columns = arr.get(0).size();
		int i = 0, j = 0;
		while (i < rows - 2) {
			j = 0;
			while (j < columns - 2) {
				int row1 = arr.get(i).get(j) + arr.get(i).get(j + 1) + arr.get(i).get(j + 2);
				int row2 = arr.get(i + 1).get(j + 1);
				int row3 = arr.get(i + 2).get(j) + arr.get(i + 2).get(j + 1) + arr.get(i + 2).get(j + 2);
				result = Math.max(result, (row1 + row2 + row3));
				j++;
			}
			i++;
		}
		return result;
	}

	public static List<Integer> rotLeft(List<Integer> a, int d) {
		int size = a.size();
		d = (d % size) * -1;
		if (a.size() == 0) {
			return a;
		} else if (d < 0) {
			d += size;
		}
		if (d == 0) {
			return a;
		}

		for (int cycleStart = 0, nMoved = 0; nMoved != size; cycleStart++) {
			Integer displaced = a.get(cycleStart);
			int i = cycleStart;
			do {
				i += d;
				if (i >= size)
					i -= size;
				displaced = a.set(i, displaced);
				nMoved++;
			} while (i != cycleStart);
		}
		return a;
	}

	public static void minimumBribes(List<Integer> q) {
		int bribe = 0;
		boolean chaotic = false;
		int n = q.size();
		for (int i = 0; i < n; i++) {
			if (q.get(i) - (i + 1) > 2) {
				chaotic = true;
				break;
			}
			for (int j = Math.max(0, q.get(i) - 2); j < i; j++) {
				if (q.get(j) > q.get(i))
					bribe++;
			}
		}
		if (chaotic)
			System.out.println("Too chaotic");
		else
			System.out.println(bribe);
	}

	static int minimumSwaps(int[] arr) {
		int swap = 0;
		for (int i = 0; i < arr.length; i++) {
			if (i + 1 != arr[i]) {
				int t = i;
				while (arr[t] != i + 1) {
					t++;
				}
				int temp = arr[t];
				arr[t] = arr[i];
				arr[i] = temp;
				swap++;
			}
		}
		return swap;
	}

	public static long arrayManipulation(int n, List<List<Integer>> queries) {
		long[] start = new long[n + 1];
		long[] end = new long[n + 1];
		long max = -1L;
		long sum = 0L;
		for (int i = 0; i < queries.size(); i++) {
			int a = queries.get(i).get(0);
			int b = queries.get(i).get(1);
			long k = queries.get(i).get(2);
			start[a] += k;
			end[b] += k;
		}
		for (int i = 1; i < n + 1; i++) {
			sum += start[i];
			if (sum > max) {
				max = sum;
			}
			sum -= end[i];
		}
		return max;
	}

	public static void checkMagazine(List<String> magazine, List<String> note) {
		if (note == null || note.isEmpty() || magazine == null || magazine.isEmpty()
				|| (note.size() > magazine.size())) {
			System.out.println("No");
			return;
		}

		HashMap<String, Integer> count = new HashMap<String, Integer>();

		for (String key : magazine) {
			if (count.containsKey(key)) {
				count.put(key, count.get(key) + 1);
			} else {
				count.put(key, 1);
			}
		}

		for (String noteValue : note) {
			if (count.containsKey(noteValue)) {
				count.put(noteValue, count.get(noteValue) - 1);
				if (count.get(noteValue) < 0) {
					System.out.println("No");
					return;
				}
			} else {
				System.out.println("No");
				return;
			}
		}
		System.out.println("Yes");
	}

	public static String twoStrings(String s1, String s2) {
		Set<Character> one = new HashSet<Character>();
		Set<Character> two = new HashSet<Character>();
		for (Character c1 : s1.toCharArray()) {
			one.add(c1);
		}
		for (Character c2 : s2.toCharArray()) {
			two.add(c2);
		}

		one.retainAll(two);
		if (!one.isEmpty()) {
			return "YES";
		}
		return "NO";
	}

	public static int sherlockAndAnagrams(String s) {
		int result = 0;
		Map<String, Integer> count = new HashMap<String, Integer>();
		for (int i = 0; i < s.length(); i++) {
			for (int j = i; j < s.length(); j++) {
				char[] array = s.substring(i, j + 1).toCharArray();
				Arrays.sort(array);
				String sub = new String(array);
				count.put(sub, count.getOrDefault(sub, 0) + 1);
			}
		}
		for (String key : count.keySet()) {
			int v = count.get(key);
			result += (v * (v - 1)) / 2;
		}
		return result;
	}

	static long countTriplets(List<Long> arr, long r) {
		Map<Long, Long> rightMap = getOccurenceMap(arr);
		Map<Long, Long> leftMap = new HashMap<>();
		long numberOfGeometricPairs = 0;

		for (long val : arr) {
			long countLeft = 0;
			long countRight = 0;
			long lhs = 0;
			long rhs = val * r;
			if (val % r == 0) {
				lhs = val / r;
			}
			Long occurence = rightMap.get(val);
			rightMap.put(val, occurence - 1L);

			if (rightMap.containsKey(rhs)) {
				countRight = rightMap.get(rhs);
			}
			if (leftMap.containsKey(lhs)) {
				countLeft = leftMap.get(lhs);
			}
			numberOfGeometricPairs += countLeft * countRight;
			insertIntoMap(leftMap, val);
		}
		return numberOfGeometricPairs;
	}

	private static Map<Long, Long> getOccurenceMap(List<Long> test) {
		Map<Long, Long> occurenceMap = new HashMap<>();
		for (long val : test) {
			insertIntoMap(occurenceMap, val);
		}
		return occurenceMap;
	}

	private static void insertIntoMap(Map<Long, Long> occurenceMap, Long val) {
		if (!occurenceMap.containsKey(val)) {
			occurenceMap.put(val, 1L);
		} else {
			Long occurence = occurenceMap.get(val);
			occurenceMap.put(val, occurence + 1L);
		}
	}

	static List<Integer> freqQuery(List<List<Integer>> queries) {
		List<Integer> ans = new ArrayList<>();
		Map<Integer, Integer> map = new HashMap<>();
		int maxFr = 0;
		for (List<Integer> ops : queries) {
			int op = ops.get(0);
			int data = ops.get(1);
			Integer v = map.get(data) == null ? 0 : map.get(data);
			if (op == 1) {
				map.put(data, ++v);
				maxFr = Math.max(maxFr, v);
			} else if (op == 2) {
				if (v > 0) {
					map.put(data, --v);
				}
			} else {
				if (data <= maxFr && map.containsValue(data))
					ans.add(1);
				else
					ans.add(0);
			}
		}
		return ans;
	}

	public static void countSwaps(List<Integer> a) {
		int n = a.size();
		int swapCount = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n - 1; j++) {
				if (a.get(j) > a.get(j + 1)) {
					swap(a, j, j + 1);
					swapCount++;
				}
			}
		}
		System.out.println("Array is sorted in " + swapCount + " swaps.");
		System.out.println("First Element: " + a.get(0));
		System.out.println("Last Element: " + a.get(n - 1));
	}

	private static void swap(List<Integer> a, int n, int m) {
		int temp = a.get(n);
		a.set(n, a.get(m));
		a.set(m, temp);
	}

	public static int maximumToys(List<Integer> prices, int k) {
		int pl = prices.size();
		int total = prices.stream().reduce(Integer::sum).get();
		if (k >= total)
			return pl;

		prices.sort(Comparator.reverseOrder());
		int reducedPrice = 0;
		for (int i = 0; i < pl; i++) {
			reducedPrice += prices.get(i);
			if (k >= total - reducedPrice) {
				return (pl - i - 1);
			}
		}

		return 0;
	}

	public static int activityNotificationsTemp(List<Integer> expenditure, int d) {
		int size = expenditure.size();
		int notifications = 0;
		if (size <= d) {
			return notifications;
		}

		for (int i = 0; i < size - d; i++) {
			int mean = expenditure.subList(i, d + i).stream().reduce(0, Integer::sum) / d;
			if ((mean * 2) > expenditure.get(d + i)) {
				notifications++;
			}
		}
		return notifications;
	}

	static int activityNotifications(List<Integer> expenditure, int d) {

		int notificationCount = 0;
		int[] data = new int[201];

		for (int i = 0; i < d; i++) {
			data[expenditure.get(i)]++;
		}

		for (int i = d; i < expenditure.size(); i++) {
			double median = getMedian(d, data);
			if (expenditure.get(i) >= 2 * median) {
				notificationCount++;

			}
			data[expenditure.get(i)]++;
			data[expenditure.get(i - d)]--;
		}
		return notificationCount;
	}

	private static double getMedian(int d, int[] data) {
		double median = 0;
		if (d % 2 == 0) {
			Integer m1 = null;
			Integer m2 = null;
			int count = 0;
			for (int j = 0; j < data.length; j++) {
				count += data[j];
				if (m1 == null && count >= d / 2) {
					m1 = j;
				}
				if (m2 == null && count >= d / 2 + 1) {
					m2 = j;
					break;
				}
			}
			median = (m1 + m2) / 2.0;
		} else {
			int count = 0;
			for (int j = 0; j < data.length; j++) {
				count += data[j];
				if (count > d / 2) {
					median = j;
					break;
				}
			}
		}
		return median;
	}

	public static long countInversions(List<Integer> arr) {
		long swapCount = 0;
		int[] helper = new int[arr.size()];
		return mergeSort(arr, 0, arr.size() - 1, helper, swapCount);
	}

	private static long mergeSort(List<Integer> arr, int l, int r, int[] helper, long swapCount) {
		if (l < r) {
			int m = (l + r) / 2;
			swapCount = mergeSort(arr, l, m, helper, swapCount) + mergeSort(arr, m + 1, r, helper, swapCount)
					+ merge(arr, l, m, r, helper, swapCount);
		}
		return swapCount;
	}

	private static long merge(List<Integer> arr, int l, int m, int r, int[] helper, long swapCount) {
		for (int i = l; i <= r; i++) {
			helper[i] = arr.get(i);
		}

		int curr = l;
		int left = l;
		int right = m + 1;

		while (left <= m && right <= r) {
			if (helper[left] <= helper[right]) {
				arr.set(curr++, helper[left++]);
			} else {
				swapCount += m + 1 - left;
				arr.set(curr++, helper[right++]);
			}
		}

		while (left <= m) {
			arr.set(curr++, helper[left++]);
		}
		return swapCount;
	}

	public static int makeAnagram(String a, String b) {
		int cArr[] = new int[26];
		for (int i = 0; i < a.length(); i++)
			cArr[a.charAt(i) - 97]++;
		for (int i = 0; i < b.length(); i++)
			cArr[b.charAt(i) - 97]--;
		int count = 0;
		for (int i = 0; i < 26; i++)
			count += Math.abs(cArr[i]);
		return count;
	}

	public static int alternatingCharacters(String s) {
		int count = 0;
		char lastItem = 0;

		for (char item : s.toCharArray()) {
			if (lastItem == item) {
				count++;
			}
			lastItem = item;
		}
		return count;
	}

	public static String isValid(String s) {
		final String YES = "YES";
		final String NO = "NO";
		if (s.isEmpty()) {
			return NO;
		} else if (s.length() <= 3) {
			return YES;
		}
		int cArr[] = new int[26];
		for (int i = 0; i < s.length(); i++)
			cArr[s.charAt(i) - 97]++;
		Arrays.sort(cArr);
		int i = 0;
		while (cArr[i] == 0) {
			i++;
		}
		int min = cArr[i];
		int max = cArr[25];

		if (min == max) {
			return YES;
		} else if ((max - min == 1) && (max > cArr[24]) || (min == 1) && (cArr[i + 1] == max)) {
			return YES;
		}
		return NO;
	}

	static long substrCount(int n, String s) {
		long res = n;
		for (int i = 0; i < n; i++) {
			boolean isAfterMedian = false;
			int beforeMedian = 1;
			char current = s.charAt(i);
			for (int j = i + 1; j < n; j++) {
				if (!isAfterMedian) {
					if (s.charAt(j) == current) {
						res++;
						beforeMedian++;
					}
					if (s.charAt(j) != current) {
						isAfterMedian = true;
					}
					;
				} else {
					if (s.charAt(j) == current && beforeMedian > 0) {
						beforeMedian--;
						if (beforeMedian == 0)
							res++;
					} else {
						break;
					}
				}
			}
		}
		return res;
	}

	public static int commonChild(String s1, String s2) {
		char[] s1_array = s1.toUpperCase().toCharArray();
		char[] s2_array = s2.toUpperCase().toCharArray();
		int[] count = new int[s1.length() + 1];

		for (int i = 1; i <= s1.length(); i++) {
			int prev = 0;
			for (int j = 1; j <= s2.length(); j++) {
				int temp = count[j];
				if (s1_array[i - 1] == s2_array[j - 1]) {
					count[j] = prev + 1;
				} else {
					count[j] = Math.max(count[j], count[j - 1]);
				}
				prev = temp;
			}
		}
		return count[s1.length()];
	}

	public static int minimumAbsoluteDifference(List<Integer> arr) {
		int[] intArr = arr.stream().sorted().mapToInt(Integer::intValue).toArray();
		int minAbsDiff = Integer.MAX_VALUE;
		for (int i = 0; i < intArr.length - 1; i++) {
			minAbsDiff = Math.min(minAbsDiff, Math.abs(intArr[i] - intArr[i + 1]));
		}
		return minAbsDiff;
	}

	public static int luckBalance(int k, List<List<Integer>> contests) {
		int size = contests.size();
		int loseCount = 0;
		int luckCount = 0;
		Collections.sort(contests, new Comparator<List<Integer>>() {
			@Override
			public int compare(List<Integer> o1, List<Integer> o2) {
				return o2.get(0).compareTo(o1.get(0));
			}
		});

		for (int i = 0; i < size; i++) {
			if (contests.get(i).get(1) == 1) {
				if (loseCount < k) {
					luckCount += contests.get(i).get(0);
					loseCount++;
				} else {
					luckCount -= contests.get(i).get(0);
				}
			} else {
				luckCount += contests.get(i).get(0);
			}
		}
		return luckCount;
	}

	static int getMinimumCost(int k, int[] c) {
		Arrays.sort(c);
		int i = c.length - 1;
		int bought = 0;
		int total = 0;
		// start backwards from the most expensive flower, stop when there is no more
		// flowers left
		while (i >= 0) {
			// Calculate total
			// increment bought by 1 when everyone in the group has bought equal number of
			// flowers
			for (int j = 0; j < k && i >= 0; j++) {
				total += (1 + bought) * c[i];
				i--;
			}
			bought++;
		}
		return total;
	}

	public static int maxMin(int k, List<Integer> arr) {
		int[] a = arr.stream().sorted().mapToInt(x -> x).toArray();
		int min = Integer.MAX_VALUE;
		for (int i = 0; i < a.length - k + 1; i++) {
			min = Math.min(min, a[i + k - 1] - a[i]);
		}
		return min;
	}

	public static String reverseShuffleMerge(String s) {
		int[] count = new int[26], used = new int[26], rem = new int[26];
		StringBuilder sb = new StringBuilder();

		for (int i = 0; i < s.length(); i++) {
			count[s.charAt(i) - 'a']++;
		}
		for (int i = 0; i < count.length; i++) {
			rem[i] = count[i];
		}

		for (int i = s.length() - 1; i >= 0; i--) {
			char c = s.charAt(i);
			if (sb.length() == 0) {
				sb.append(c);
				used[c - 'a']++;
			} else {
				if (2 * used[c - 'a'] == count[c - 'a']) {
					rem[c - 'a']--;
					continue;
				}

				while (sb.length() > 0) {
					char last = sb.charAt(sb.length() - 1);
					if (c < last && 2 * (rem[last - 'a'] + used[last - 'a']) > count[last - 'a']) {
						used[last - 'a']--;
						sb.deleteCharAt(sb.length() - 1);
					} else {
						break;
					}
				}

				sb.append(c);
				used[c - 'a']++;
			}
			rem[c - 'a']--;
		}
		return sb.toString();
	}

	public static void whatFlavors(List<Integer> cost, int money) {
		Map<Integer, Integer> hash = new HashMap<Integer, Integer>();

		int[] res = new int[2];

		for (int i = 0; i < cost.size(); i++) {
			hash.put(cost.get(i), i);
		}

		for (int i = 0; i < cost.size(); i++) {
			int k = money - cost.get(i);
			if (hash.containsKey(k) && hash.get(k) != i) {
				res[0] = i + 1;
				res[1] = hash.get(k) + 1;
				break;
			}
		}

		System.out.printf("%d %d%n", res[0], res[1]);
	}

	public static int pairs(int k, List<Integer> arr) {
		int count;
		HashSet<Integer> set = new HashSet<>(arr);
		count = (int) arr.stream().filter(number -> set.contains(number + k)).count();
		return count;
	}

	static long triplets(int[] a, int[] b, int[] c) {

		long count = 0;

		a = Arrays.stream(a).sorted().distinct().toArray();
		b = Arrays.stream(b).sorted().distinct().toArray();
		c = Arrays.stream(c).sorted().distinct().toArray();

		for (int i = 0; i < b.length; i++) {
			int c1 = getIndex(a, b[i]) + 1;
			int c2 = getIndex(c, b[i]) + 1;
			if (c1 == -1 || c2 == -1)
				continue;
			count += (long) c1 * (long) c2;
		}
		return count;
	}

	public static int getIndex(int[] arr, int e) {
		int index = -1;
		int l = 0, r = arr.length - 1;
		while (l <= r) {
			int mid = (l + r) / 2;
			if (arr[mid] <= e) {
				if (arr[mid] == e) {
					index = mid;
					break;
				}
				index = mid;
				l = mid + 1;
			} else
				r = mid - 1;
		}
		return index;
	}

	static long minTime(long[] machines, long goal) {
		Arrays.sort(machines);
		long max = (machines[machines.length - 1]) * goal;
		long min = 0;
		long result = -1;
		while (max > min) {
			long midValue = (max + min) / 2;
			long unit = 0;
			for (long machine : machines) {
				unit += midValue / machine;
			}
			if (unit < goal) {
				min = midValue + 1;
				result = midValue + 1;
			} else {
				max = midValue;
				result = midValue;
			}
		}
		return result;
	}

	public static long maximumSum(List<Long> a, long m) {
		long max = 0, curr = 0;
		TreeSet<Long> t = new TreeSet<>();
		for (int i = 0; i < a.size(); i++) {
			curr = (curr + a.get(i) % m) % m;
			max = Math.max(max, curr);
			Long p = t.higher(curr);
			if (p != null) {
				max = Math.max(max, (curr - p + m) % m);
			}
			t.add(curr);
		}
		return max;
	}

	public static long minimumPasses(long m, long w, long p, long n) {
		long candies = 0;
		long invest = 0;
		long spend = Long.MAX_VALUE;

		while (candies < n) {
			long passes = (long) (((p - candies) / (double) m) / w);
			if (passes <= 0) {
				long mw = candies / p + m + w;
				long half = mw >>> 1;
				if (m > w) {
					m = Math.max(m, half);
					w = mw - m;
				} else {
					w = Math.max(w, half);
					m = mw - w;
				}
				candies %= p;
				passes++;
			}

			long mw;
			long pmw;
			try {
				mw = Math.multiplyExact(m, w);
				pmw = Math.multiplyExact(passes, mw);
			} catch (ArithmeticException ex) {
				invest += 1;
				spend = Math.min(spend, invest + 1);
				break;
			}
			candies += pmw;
			invest += passes;
			long increment = (long) Math.ceil((n - candies) / (double) mw);
			spend = Math.min(spend, invest + increment);
		}
		return Math.min(spend, invest);
	}

	static int maxSubsetSum(int[] arr) {
		if (arr.length == 0)
			return 0;
		arr[0] = Math.max(0, arr[0]);
		if (arr.length == 1)
			return arr[0];
		arr[1] = Math.max(arr[0], arr[1]);
		for (int i = 2; i < arr.length; i++)
			arr[i] = Math.max(arr[i - 1], arr[i] + arr[i - 2]);
		return arr[arr.length - 1];
	}

	public static String abbreviation(String a, String b) {
		boolean[][] dp = new boolean[b.length() + 1][a.length() + 1];
		dp[0][0] = true;

		for (int j = 1; j < dp[0].length; j++) {
			if (Character.isLowerCase(a.charAt(j - 1)))
				dp[0][j] = dp[0][j - 1];
		}

		for (int i = 1; i < dp.length; i++) {
			for (int j = 1; j < dp[0].length; j++) {
				char ca = a.charAt(j - 1), cb = b.charAt(i - 1);
				if (ca >= 'A' && ca <= 'Z') {
					if (ca == cb) {
						dp[i][j] = dp[i - 1][j - 1];
					}
				} else {
					ca = Character.toUpperCase(ca);
					if (ca == cb)
						dp[i][j] = dp[i - 1][j - 1] || dp[i][j - 1];
					else
						dp[i][j] = dp[i][j - 1];
				}
			}
		}

		return dp[b.length()][a.length()] ? "YES" : "NO";

	}

	public static long candies(int n, List<Integer> arr) {
		long result = n;
		int len = arr.size();
		int[] candies = new int[len];
		for (int i = 0; i < len - 1; i++) {
			if (arr.get(i + 1) > arr.get(i) && !(candies[i + 1] > candies[i])) {
				candies[i + 1] = candies[i] + 1;
			}
		}
		for (int j = len - 1; j > 0; j--) {
			if (arr.get(j) < arr.get(j - 1) && !(candies[j] < candies[j - 1])) {
				candies[j - 1] = candies[j] + 1;
			}
		}
		for (int k = 0; k < len; k++) {
			result += candies[k];
		}
		return result;
	}

	public static String isBalanced(String s) {
		HashMap<Character, Character> brackets = new HashMap<>();
		brackets.put('(', ')');
		brackets.put('[', ']');
		brackets.put('{', '}');
		Deque<Character> stack = new LinkedList<>();

		for (Character c : s.toCharArray()) {
			if (brackets.containsKey(c))
				stack.push(c);
			else if (!c.equals(brackets.get(stack.poll())))
				return "NO";
		}
		return stack.isEmpty() ? "YES" : "NO";
	}

	public static long largestRectangle(List<Integer> h) {
		Stack<Integer> SK = new Stack<Integer>();
		int max_area = 0;
		int tp;
		int area_with_top;
		int i = 0;
		while (i < h.size()) {
			if (SK.empty() || h.get(SK.peek()) <= h.get(i)) {
				SK.push(i);
				i = i + 1;
			} else {
				tp = SK.peek();
				SK.pop();
				area_with_top = h.get(tp) * (SK.empty() ? i : i - SK.peek() - 1);
				if (max_area < area_with_top)
					max_area = area_with_top;
			}
		}

		while (SK.empty() == false) {
			tp = SK.peek();
			SK.pop();
			area_with_top = h.get(tp) * (SK.empty() ? i : i - SK.peek() - 1);
			if (max_area < area_with_top)
				max_area = area_with_top;
		}

		return (max_area);
	}

	static long[] riddle(long[] arr) {
		int n = arr.length;
		Stack<Integer> st = new Stack<>();
		int[] left = new int[n + 1];
		int[] right = new int[n + 1];

		for (int i = 0; i < n; i++) {
			left[i] = -1;
			right[i] = n;
		}

		for (int i = 0; i < n; i++) {
			while (!st.isEmpty() && arr[st.peek()] >= arr[i])
				st.pop();

			if (!st.isEmpty())
				left[i] = st.peek();

			st.push(i);
		}
		while (!st.isEmpty()) {
			st.pop();
		}

		for (int i = n - 1; i >= 0; i--) {
			while (!st.isEmpty() && arr[st.peek()] >= arr[i])
				st.pop();

			if (!st.isEmpty())
				right[i] = st.peek();

			st.push(i);
		}

		long ans[] = new long[n + 1];

		for (int i = 0; i <= n; i++) {
			ans[i] = 0;
		}

		for (int i = 0; i < n; i++) {
			int len = right[i] - left[i] - 1;
			ans[len] = Math.max(ans[len], arr[i]);
		}

		for (int i = n - 1; i >= 1; i--) {
			ans[i] = Math.max(ans[i], ans[i + 1]);
		}

		long[] res = new long[n];

		for (int i = 1; i <= n; i++) {
			res[i - 1] = ans[i];
		}

		return res;

	}

}
