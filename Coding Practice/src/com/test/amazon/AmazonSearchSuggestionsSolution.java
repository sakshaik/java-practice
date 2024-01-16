package com.test.amazon;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Result {

	public static List<List<String>> searchSuggestions(List<String> repository, String customerQuery) {
		Node root = new Node();
		for (String repo : repository) {
			addWord(repo, root);
		}
		String searching = "";
		List<List<String>> result = new ArrayList<>();
		for (int i = 0; i < customerQuery.length(); i++) {
			searching += Character.toString(customerQuery.charAt(i));
			if (i != 0) {
				result.add(search(searching, root, new ArrayList<>()));
			}
		}
		return result;
	}

	public static void addWord(String s, Node root) {
		Node current = root;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (current.next[c - 'a'] == null) {
				current.next[c - 'a'] = new Node();
			}
			current.wordCount++;
			current = current.next[c - 'a'];
		}
		current.word = s;
		current.end = true;
	}

	public static List<String> search(String s, Node root, List<String> output) {
		Node current = root;
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

	public static void dfs(Node current, List<String> output) {
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
}

class Node {
	Node[] next = new Node[26];
	boolean end;
	String word;

	Node() {
		for (int i = 0; i < 26; i++) {
			next[i] = null;
		}
		end = false;
		word = "";
	}

	int wordCount = 0;
}

public class AmazonSearchSuggestionsSolution {
	public static void main(String[] args) throws IOException {
		System.out
				.println(Result.searchSuggestions(Arrays.asList("bags", "baggage", "banner", "box", "cloths"), "bags"));
	}
}
