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
    private PartTwoLinkedList service;

    @Test
    void leetCode203Test() {
        PartTwoLinkedList.ListNode listNode7 = new PartTwoLinkedList.ListNode(6, null);
        PartTwoLinkedList.ListNode listNode6 = new PartTwoLinkedList.ListNode(5, listNode7);
        PartTwoLinkedList.ListNode listNode5 = new PartTwoLinkedList.ListNode(4, listNode6);
        PartTwoLinkedList.ListNode listNode4 = new PartTwoLinkedList.ListNode(3, listNode5);
        PartTwoLinkedList.ListNode listNode3 = new PartTwoLinkedList.ListNode(6, listNode4);
        PartTwoLinkedList.ListNode listNode2 = new PartTwoLinkedList.ListNode(2, listNode3);
        PartTwoLinkedList.ListNode head = new PartTwoLinkedList.ListNode(1, listNode2);

        PartTwoLinkedList.ListNode result = service.leetCode203(head, 6);
        System.out.println(result.toString());
    }

    @Test
    void leetCode24Test() {
        PartTwoLinkedList.ListNode listNode4 = new PartTwoLinkedList.ListNode(4, null);
        PartTwoLinkedList.ListNode listNode3 = new PartTwoLinkedList.ListNode(3, listNode4);
        PartTwoLinkedList.ListNode listNode2 = new PartTwoLinkedList.ListNode(2, listNode3);
        PartTwoLinkedList.ListNode head = new PartTwoLinkedList.ListNode(1, listNode2);

        PartTwoLinkedList.ListNode result = service.leetCode24(head);
        System.out.println(result.toString());
    }

    @Test
    void leetCode25Test() {
        PartTwoLinkedList.ListNode listNode7 = new PartTwoLinkedList.ListNode(7, null);
        PartTwoLinkedList.ListNode listNode6 = new PartTwoLinkedList.ListNode(6, listNode7);
        PartTwoLinkedList.ListNode listNode5 = new PartTwoLinkedList.ListNode(5, listNode6);
        PartTwoLinkedList.ListNode listNode4 = new PartTwoLinkedList.ListNode(4, listNode5);
        PartTwoLinkedList.ListNode listNode3 = new PartTwoLinkedList.ListNode(3, listNode4);
        PartTwoLinkedList.ListNode listNode2 = new PartTwoLinkedList.ListNode(2, listNode3);
        PartTwoLinkedList.ListNode head = new PartTwoLinkedList.ListNode(1, listNode2);

        PartTwoLinkedList.ListNode result = service.leetCode25(head, 2);
        while (result != null) {
            System.out.println(result.val);
            result = result.next;
        }
    }

    @Test
    void leetCode143Test() {
//        PartTwoLinkedList.ListNode listNode7 = new PartTwoLinkedList.ListNode(7, null);
//        PartTwoLinkedList.ListNode listNode6 = new PartTwoLinkedList.ListNode(6, listNode7);
//        PartTwoLinkedList.ListNode listNode5 = new PartTwoLinkedList.ListNode(5, listNode6);
        PartTwoLinkedList.ListNode listNode4 = new PartTwoLinkedList.ListNode(4, null);
        PartTwoLinkedList.ListNode listNode3 = new PartTwoLinkedList.ListNode(3, listNode4);
        PartTwoLinkedList.ListNode listNode2 = new PartTwoLinkedList.ListNode(2, listNode3);
        PartTwoLinkedList.ListNode head = new PartTwoLinkedList.ListNode(1, listNode2);

        service.leetCode143(head);
        System.out.println(head);
    }

    @Test
    void leetCode19Test() {
        PartTwoLinkedList.ListNode listNode5 = new PartTwoLinkedList.ListNode(5, null);
        PartTwoLinkedList.ListNode listNode4 = new PartTwoLinkedList.ListNode(4, listNode5);
        PartTwoLinkedList.ListNode listNode3 = new PartTwoLinkedList.ListNode(3, listNode4);
        PartTwoLinkedList.ListNode listNode2 = new PartTwoLinkedList.ListNode(2, listNode3);
        PartTwoLinkedList.ListNode head = new PartTwoLinkedList.ListNode(1, listNode2);

        PartTwoLinkedList.ListNode result = service.leetCode19(head, 2);
        System.out.println(result);
    }

}
