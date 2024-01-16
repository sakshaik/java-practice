package com.test.euler;

import java.util.Scanner;

// https://projecteuler.net/problem=3
public class Problem3 {

	public static void main(String[] args) {
		Scanner number = new Scanner(System.in);
		System.out.println("Enter no to find its largest prime factor");
		Long value = Long.valueOf(number.nextLine());
		System.out.println(findLargestPrimeFactor(value));
		number.close();
	}

	private static Long findLargestPrimeFactor(Long value) {
		Long i = 1L;
		for (i = 2L; i < value; i++) {
			if (value % i == 0) {
				value /= i;
				i--;
			}
		}
		return i;
	}

}
