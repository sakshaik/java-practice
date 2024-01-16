package com.test.collections;

import java.util.HashMap;
import java.util.Map;

public class HashMapTest {

	public static void main(String[] args) {
		Map<Character, Integer> count = new HashMap<Character, Integer>();
		String a = "abc";
		String b = "cda";
		int result = 0;

		for (int i = 0; i < a.length(); i++) {
			if (count.containsKey(a.charAt(i))) {
				count.put(a.charAt(i), count.get(a.charAt(i)) + 1);
			} else {
				count.put(a.charAt(i), 1);
			}
		}

		for (int j = 0; j < b.length(); j++) {
			if (count.containsKey(b.charAt(j))) {
				count.put(b.charAt(j), count.get(b.charAt(j)) + 1);
			} else {
				count.put(b.charAt(j), 1);
			}
		}

		for (Character c : count.keySet()) {
			if (count.get(c) <= 1) {
				result++;
			}
		}
		System.out.println(result);
	}

}
