package com.test.amazon;

import java.util.Arrays;
import java.util.List;

public class AmazonFreshPromoSolution {

	private boolean process(List<List<String>> codeList, List<String> shoppingList) {
		if (codeList == null || codeList.isEmpty()) {
			return true;
		}
		if (shoppingList == null || shoppingList.isEmpty()) {
			return false;
		}

		int i = 0, j = 0;
		while (i < codeList.size() && j + codeList.size() <= shoppingList.size()) {
			boolean match = true;
			for (int k = 0; k < codeList.size(); k++) {
				if (!codeList.get(i).get(k).equalsIgnoreCase("anything")
						&& !shoppingList.get(j + k).equalsIgnoreCase(codeList.get(i).get(k))) {
					match = false;
					break;
				}
			}
			if (match) {
				j += codeList.get(i).size();
				i++;
			} else {
				j++;
			}
		}
		return i == codeList.size() ? true : false;
	}

	public static void main(String[] args) {
		AmazonFreshPromoSolution solution = new AmazonFreshPromoSolution();
		List<List<String>> codeList = Arrays.asList(Arrays.asList("apple", "apple"),
				Arrays.asList("banana", "anything", "banana"));
		List<String> shoppingList = Arrays.asList("orange", "apple", "apple", "banana", "orange", "banana");
		System.out.println(solution.process(codeList, shoppingList));
		
		List<List<String>> codeList2 = Arrays.asList(Arrays.asList("banana", "apple"),
				Arrays.asList("banana", "anything", "banana"));
		List<String> shoppingList2 = Arrays.asList("orange", "apple", "apple", "banana", "orange", "banana");
		System.out.println(solution.process(codeList2, shoppingList2));
	}

}
