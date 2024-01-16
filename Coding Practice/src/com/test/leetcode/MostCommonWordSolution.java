package com.test.leetcode;

public class MostCommonWordSolution {

	// Most Common Word
	/*
	 * Given a string paragraph and a string array of the banned words banned,
	 * return the most frequent word that is not banned. It is guaranteed there is
	 * at least one word that is not banned, and that the answer is unique.
	 * 
	 * The words in paragraph are case-insensitive and the answer should be returned
	 * in lowercase.
	 */

	public static void main(String[] args) {
		MostCommonWordSolution solution = new MostCommonWordSolution();
		String[] banned = { "hit" };
		System.out.println(solution.mostCommonWord("Bob hit a ball, the hit BALL flew far after it was hit.", banned));
		String[] banned2 = { "ban", "bad" };
		System.out.println(solution.mostCommonWord("I will ban bad calls, if all calls are bad", banned2));
	}

	public String mostCommonWord(String paragraph, String[] banned) {
		Trie trie = new Trie();
		for (String w : banned)
			trie.add(0, w, true);

		int idx = 0;
		while (idx < paragraph.length())
			idx = trie.add(idx, paragraph, false);

		if (trie.rootMaxCount == null)
			return null;

		return paragraph.substring(trie.rootMaxCount.wordStartIdx, trie.rootMaxCount.wordEndIdx + 1).toLowerCase();
	}

	static final class Trie {
		Trie rootMaxCount;
		boolean terminal;
		boolean banned;
		int count;
		int wordStartIdx;
		int wordEndIdx;
		Trie[] children;

		int add(int start, String text, boolean ban) {
			if (!Character.isAlphabetic(text.charAt(start)))
				return start + 1;

			int idx = start;
			Trie current = this;
			while (idx < text.length() && Character.isAlphabetic(text.charAt(idx))) {
				int lc = Character.toLowerCase(text.charAt(idx++)) - 'a';
				if (current.children == null)
					current.children = new Trie[26];

				Trie next = current.children[lc];
				if (next == null) {
					current.children[lc] = new Trie();
					next = current.children[lc];
				}
				current = next;
			}

			current.terminal = true;
			current.banned = current.banned || ban;
			current.count++;
			current.wordStartIdx = start;
			current.wordEndIdx = idx - 1;

			if (!current.banned && (rootMaxCount == null || rootMaxCount.count < current.count))
				rootMaxCount = current;

			return idx;
		}
	}

}
