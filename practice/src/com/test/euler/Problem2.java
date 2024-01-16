package com.test.euler;

import java.util.Scanner;

public class Problem2 {

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		System.out.println("Enter limit of Fibonacci Terms:");
		Long limit = Long.valueOf(scanner.nextLine());
		System.out.println(findSumOfAllEvenFibonacci(limit));
		scanner.close();
	}

	private static Long findSumOfAllEvenFibonacci(Long limit) {
		Long t1 = 0L;
		Long t2 = 1L;
		Long sum = 0L;
		Long total = 0L;
		while (sum <= limit) {
			sum = t1 + t2;
			t1 = t2;
			t2 = sum;
			if (sum % 2 == 0) {
				total += sum;
			}
		}
		return total;
	}

}
