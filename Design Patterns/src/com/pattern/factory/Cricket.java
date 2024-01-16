package com.pattern.factory;

public class Cricket implements Sport {

	@Override
	public void substitutes() {
		System.out.println("Cricket has 0 subs");
	}

}
