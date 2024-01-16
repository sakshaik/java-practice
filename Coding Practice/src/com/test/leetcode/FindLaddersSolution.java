package com.test.leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

public class FindLaddersSolution {

	// Word Ladder II
	/*
	 * A transformation sequence from word beginWord to word endWord using a
	 * dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk
	 * such that:
	 * 
	 * Every adjacent pair of words differs by a single letter. Every si for 1 <= i
	 * <= k is in wordList. Note that beginWord does not need to be in wordList. sk
	 * == endWord Given two words, beginWord and endWord, and a dictionary wordList,
	 * return all the shortest transformation sequences from beginWord to endWord,
	 * or an empty list if no such sequence exists. Each sequence should be returned
	 * as a list of the words [beginWord, s1, s2, ..., sk]
	 */

	public static void main(String[] args) {
		FindLaddersSolution solution = new FindLaddersSolution();
		System.out.println(solution.findLadders("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log", "cog")));
		System.out.println(solution.findLadders("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log")));
		System.out
				.println(solution.ladderLength("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log", "cog")));
		System.out.println(solution.ladderLength("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log")));
	}

	public int ladderLength(String beginWord, String endWord, List<String> wordList) {
		List<List<String>> laddersList = findLadders(beginWord, endWord, wordList);
		return laddersList.isEmpty() ? 0 : laddersList.stream().mapToInt(List::size).max().getAsInt();
	}

	public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
		List<List<String>> result = new ArrayList<>();
		Set<String> set = new HashSet<>();
		Queue<CharacterNode> q = new LinkedList<>();

		for (String s : wordList)
			set.add(s); // add wordList to set

		// if endWord is not in wordList return blank result as no possible path
		if (!set.contains(endWord))
			return result;

		q.add(new CharacterNode(beginWord)); // add the beginCharacterNode

		while (!q.isEmpty()) {
			int size = q.size();
			// tempset to remove the used word all together after every Iteration
			Set<String> removeSet = new HashSet<>();
			for (int i = 0; i < size; i++) {
				CharacterNode cur = q.poll();
				if (cur.name.equals(endWord)) {
					result.add(cur.path); // match found add the path history to the result
				} else {
					List<String> neighbours = getNeighbours(cur.name, set);
					for (String n : neighbours) {
						q.add(new CharacterNode(n, cur.path));
						removeSet.add(n); // add the words getting used, later we will delete all.
					}
				}
			}
			set.removeAll(removeSet); // remove the words used in this traversal
		}

		return result;
	}

	private List<String> getNeighbours(String word, Set<String> set) {
		char[] ch = word.toCharArray();
		List<String> words = new ArrayList<>();
		// replace each char with from a to z
		// and check if thats a valid word
		// if valid add to neighbours list
		for (int i = 0; i < ch.length; i++) {
			char temp = ch[i];
			for (char j = 'a'; j <= 'z'; j++) {
				ch[i] = j;
				String newWord = new String(ch);
				if (set.contains(newWord))
					words.add(newWord);
			}
			ch[i] = temp;
		}
		return words;
	}

}

class CharacterNode {
	String name;
	LinkedList<String> path;

	// add the string word as name & add it to path as well
	public CharacterNode(String name) {
		this.name = name;
		this.path = new LinkedList<>();
		this.path.add(name);
	}

	// add the name, add path history from parent and then add the current as well.
	public CharacterNode(String name, LinkedList<String> path) {
		this.name = name;
		this.path = new LinkedList<>();
		this.path.addAll(path);
		this.path.add(name);
	}
}
