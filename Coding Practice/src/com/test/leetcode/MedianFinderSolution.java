package com.test.leetcode;

import java.util.Collections;
import java.util.PriorityQueue;

public class MedianFinderSolution {
	public static void main(String[] args) {
		MedianFinder medianFinder = new MedianFinder();
		medianFinder.addNum(1);
		medianFinder.addNum(2);
		medianFinder.addNum(3);
		System.out.println(medianFinder.findMedian());
		medianFinder.addNum(3);
		System.out.println(medianFinder.findMedian());
	}

}

class MedianFinder {

	private PriorityQueue<Integer> minHeap;
	private PriorityQueue<Integer> maxHeap;

	public MedianFinder() {
		minHeap = new PriorityQueue<>();
		maxHeap = new PriorityQueue<>(Collections.reverseOrder());
	}

	public void addNum(int num) {
		maxHeap.offer(num);
		minHeap.offer(maxHeap.poll());
		if (maxHeap.size() < minHeap.size())
			maxHeap.offer(minHeap.poll());
	}

	public double findMedian() {
		if (maxHeap.size() == minHeap.size())
			return (maxHeap.peek() + minHeap.peek()) / 2.0;
		else
			return maxHeap.peek();
	}
}