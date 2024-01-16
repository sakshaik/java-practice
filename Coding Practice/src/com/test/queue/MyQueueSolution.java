package com.test.queue;

import java.util.Scanner;
import java.util.Stack;

public class MyQueueSolution {
	public static void main(String[] args) {
		MyQueue<Integer> queue = new MyQueue<Integer>();

		Scanner scan = new Scanner(System.in);
		int n = scan.nextInt();

		for (int i = 0; i < n; i++) {
			int operation = scan.nextInt();
			if (operation == 1) { // enqueue
				queue.enqueue(scan.nextInt());
			} else if (operation == 2) { // dequeue
				queue.dequeue();
			} else if (operation == 3) { // print/peek
				System.out.println(queue.peek());
			}
		}
		scan.close();
	}

	static class MyQueue<T> {
		private Stack<T> reversed;
		private Stack<T> normal;

		public MyQueue() {
			normal = new Stack<>();
			reversed = new Stack<>();
		}

		public void enqueue(T item) {
			reversed.push(item);
		}

		private void pour() {
			while (!reversed.isEmpty()) {
				normal.push(reversed.pop());
			}
		}

		public T peek() {
			if (normal.isEmpty()) {
				pour();
			}
			return normal.peek();
		}

		public T dequeue() {
			if (normal.isEmpty()) {
				pour();
			}
			return normal.pop();
		}
	}

}
