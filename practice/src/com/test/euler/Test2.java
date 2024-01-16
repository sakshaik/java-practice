package com.test.euler;

import java.text.ParseException;

public class Test2 {

	public static void main(String[] args) throws ParseException {
		System.out.println(isAlphaNumeric("3a3a"));
	}

	private static boolean isAlphaNumeric(String iataNumber) {
		 return iataNumber != null && iataNumber.matches("^[0-9]*$");
	}

}
