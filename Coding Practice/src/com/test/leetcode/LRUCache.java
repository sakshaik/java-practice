package com.test.leetcode;

import java.util.HashMap;

public class LRUCache {

	HashMap<Integer, LRUNode> map;
	int capacity, count;
	LRUNode head, tail;

	public LRUCache(int capacity) {
		this.capacity = capacity;
		map = new HashMap<>();
		head = new LRUNode(0, 0);
		tail = new LRUNode(0, 0);
		head.next = tail;
		tail.pre = head;
		head.pre = null;
		tail.next = null;
		count = 0;
	}

	public void deleteNode(LRUNode node) {
		node.pre.next = node.next;
		node.next.pre = node.pre;
	}

	public void addToHead(LRUNode node) {
		node.next = head.next;
		node.next.pre = node;
		node.pre = head;
		head.next = node;
	}

	public int get(int key) {
		if (map.get(key) != null) {
			LRUNode node = map.get(key);
			int result = node.value;
			deleteNode(node);
			addToHead(node);
			return result;
		}
		return -1;
	}

	public void put(int key, int value) {
		if (map.get(key) != null) {
			LRUNode node = map.get(key);
			node.value = value;
			deleteNode(node);
			addToHead(node);
		} else {
			LRUNode node = new LRUNode(key, value);
			map.put(key, node);
			if (count < capacity) {
				count++;
				addToHead(node);
			} else {
				map.remove(tail.pre.key);
				deleteNode(tail.pre);
				addToHead(node);
			}
		}
	}
}

class LRUNode {
	int key;
	int value;
	LRUNode pre;
	LRUNode next;

	public LRUNode(int key, int value) {
		this.key = key;
		this.value = value;
	}
}