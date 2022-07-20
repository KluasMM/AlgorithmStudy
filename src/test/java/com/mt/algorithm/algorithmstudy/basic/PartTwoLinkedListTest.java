package com.mt.algorithm.algorithmstudy.basic;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * @Description
 * @Author T
 * @Date 2022/7/18
 */
@SpringBootTest
public class PartTwoLinkedListTest {

    @Autowired
    private PartTwoLinkedList partTwoLinkedList;

    @Test
    void leetCode203Test() {
        PartTwoLinkedList.ListNode listNode7 = new PartTwoLinkedList.ListNode(6, null);
        PartTwoLinkedList.ListNode listNode6 = new PartTwoLinkedList.ListNode(5, listNode7);
        PartTwoLinkedList.ListNode listNode5 = new PartTwoLinkedList.ListNode(4, listNode6);
        PartTwoLinkedList.ListNode listNode4 = new PartTwoLinkedList.ListNode(3, listNode5);
        PartTwoLinkedList.ListNode listNode3 = new PartTwoLinkedList.ListNode(6, listNode4);
        PartTwoLinkedList.ListNode listNode2 = new PartTwoLinkedList.ListNode(2, listNode3);
        PartTwoLinkedList.ListNode head = new PartTwoLinkedList.ListNode(1, listNode2);

        PartTwoLinkedList.ListNode result = partTwoLinkedList.leetCode203(head, 6);
        System.out.println(result.toString());
    }

    @Test
    void leetCode24Test() {
        PartTwoLinkedList.ListNode listNode4 = new PartTwoLinkedList.ListNode(4, null);
        PartTwoLinkedList.ListNode listNode3 = new PartTwoLinkedList.ListNode(3, listNode4);
        PartTwoLinkedList.ListNode listNode2 = new PartTwoLinkedList.ListNode(2, listNode3);
        PartTwoLinkedList.ListNode head = new PartTwoLinkedList.ListNode(1, listNode2);

        PartTwoLinkedList.ListNode result = partTwoLinkedList.leetCode24(head);
        System.out.println(result.toString());
    }

}
