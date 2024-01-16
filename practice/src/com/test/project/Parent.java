package com.test.project;

public class Parent {

	private Status status;

	public Status getStatus() {
		if (null == status) {
			status = new Status();
		}
		return status;
	}

	public void setStatus(Status status) {
		this.status = status;
	}

	public static void main(String[] args) {
		Parent parent = new Parent();
		if (parent.getStatus().getErrors() != null && parent.getStatus().getErrors().size() > 0) {
			System.out.println(true);
		}
	}

}
