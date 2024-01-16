package com.test.euler;

public class Problem24 {

	public static void main(String[] args) {
		
		int[] factorials = getDigitFactorials();

		long left = 1000000;
		String answer = "";

		while (answer.length() < 10) {
			for (int i = 0; i < 10; i++) {
				if (!answer.contains(Integer.toString(i))) {
					if (left - factorials[9 - answer.length()] > 0) {
						left = left - factorials[9 - answer.length()];
					} else {
						answer += Integer.toString(i);
						break;
					}
				}
			}
		}

		System.out.println(answer);

	}

	public static int[] getDigitFactorials() {
		int[] retVal = new int[10];
		for (int i = 0; i < retVal.length; i++)
			retVal[i] = (int) getFactorial(i);

		return retVal;
	}

	public static long getFactorial(int num) {
		long retVal = 1;

		for (int i = 2; i <= num; i++)
			retVal *= i;

		return retVal;
	}

}