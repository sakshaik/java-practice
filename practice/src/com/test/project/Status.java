package com.test.project;

import java.util.ArrayList;
import java.util.List;

public class Status {

	private List<Errors> errors;

	public List<Errors> getErrors() {
		if (null == errors) {
			errors = new ArrayList<Errors>();
		}
		return errors;
	}

	public void setErrors(List<Errors> errors) {
		this.errors = errors;
	}

}
