package com.test.euler;

import java.util.ArrayList;
import java.util.List;

public class Problem10 {

	private static List<Long> listOfPrimes = new ArrayList<Long>();
	private static int limit = 2000000;

	public static void main(String args[]) {
		long count = 0;
		for (long i = 2; i < limit; i++) {
			if (isPrime(i)) {
				count += i;
			}
		}
		System.out.println("Total " + count);
	}

	private static boolean isPrime(long n) {
		String strFromN = Long.toString(n);
		if ((strFromN.length() != 1) && (strFromN.endsWith("2") || strFromN.endsWith("4") || strFromN.endsWith("5")
				|| strFromN.endsWith("6") || strFromN.endsWith("8"))) {
			return false;
		}
		for (Long num : listOfPrimes) {
			if (num > Math.sqrt(n)) {
				break;
			}
			if (n % num.longValue() == 0) {
				return false;
			}
		}
		listOfPrimes.add(n);
		return true;
	}
}
