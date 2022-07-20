package com.mt.algorithm.algorithmstudy.basic;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.springframework.stereotype.Service;

/**
 * @Description
 * @Author T
 * @Date 2022/7/18
 */
@Service
public class PartTwoLinkedList {

    /**
     * Definition for singly-linked list.
     */
    @AllArgsConstructor
    @NoArgsConstructor
    @ToString
    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int val) {
            this.val = val;
        }
    }

    /**
     * 203. 移除链表元素
     * <p>
     * 给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。
     * <p>
     * 解题分析：
     * 区分为删除头结点和非头结点
     * 头结点：head = head.next
     * 非头结点：node.pre.next = node.next
     * 所以：添加一个虚拟头结点dummyHead 保持每个节点的删除规则都是 node.pre.next = node.next
     * 同时注意：currentHead要为删除节点的上一个节点 因为是个单向链表
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode leetCode203(ListNode head, int val) {
        /*
         * 解题分析：虚拟头结点
         *  区分为删除头结点和非头结点
         *   头结点：head = head.next
         *   非头结点：node.pre.next = node.next
         *  所以：添加一个虚拟头结点dummyHead 保持每个节点的删除规则都是 node.pre.next = node.next
         *  同时注意：currentHead要为删除节点的上一个节点 因为是个单向链表
         */
//        if (head == null) {
//            return null;
//        }
//
//        ListNode dummyHead = new ListNode(0, head);
//        ListNode pre = dummyHead;
//        ListNode current = pre.next;
//
//        while (current != null) {
//            if (current.val == val) {
//                pre.next = current.next;
//            } else {
//                pre = current;
//            }
//            current = current.next;
//        }
//
//        return dummyHead.next;

        /*
         * 解题分析2：递归
         *  首先要明确 方法的返回值其实是入参结点 即入参等于出参 除了要删除的结点 要删除的结点返回的是结点的下一个结点
         *  那么将下一个结点依次压入方法栈 每次返回的也就是压入的结点 即head.next = leetCode203(head.next, val);
         *  head为当前结点 如果当前结点为要删除元素 那么就返回下一个结点 否则就返回当前结点 即入参等于出参
         *  注：执行顺序为后序遍历 即从最后一个结点依次处理返回逻辑
         */
        if (head == null) {
            return null;
        }

        //依次将node压入方法栈 返回值为当前node的头结点 即入参等于出参
        head.next = leetCode203(head.next, val);

        //此步理解为二叉树的后续遍历 最先到达这步的是最后一个元素
        //如果当前需要删除 则跳过该结点返回下一个结点
        if (head.val == val) {
            return head.next;
        }

        //如果不是需要删除的元素 则返回当前结点
        return head;
    }

    /**
     * 206. 反转链表
     * <p>
     * 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
     *
     * @param head
     * @return
     */
    public ListNode leetCode206(ListNode head) {
        /*
         * 解题思路1：尾递归
         *  初始化pre=null cur=head temp = cur.next
         *  用temp临时存储结点的next数据 然后将结点的next指向pre 依次执行
         */
        /*return reverse(null, head);*/

        /*
         * 解题思路2：自身递归
         */
        /*if(head == null || head.next == null) {
            return head;
        }

        ListNode result = leetCode206(head.next);

        head.next.next = head;
        head.next = null;

        return result;*/

        /*
         * 解题思路3：双指针 类似于解题思路1 尾递归
         *  定义一个前置结点pre 当前结点cur的next先暂存temp 然后将cur的next指向pre
         *  然后pre指向当前cur cur指向temp
         *  直到cur为null时 前置结点pre就是结果
         */
        ListNode pre = null;
        ListNode cur = head;

        while (cur != null) {
            ListNode temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }

        return pre;
    }

    private ListNode reverse(ListNode pre, ListNode current) {
        if (current == null) {
            return pre;
        }

        ListNode temp = current.next;
        current.next = pre;
        return reverse(current, temp);
    }

    /**
     * 24. 两两交换链表中的节点
     * <p>
     * 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
     *
     * @param head
     * @return
     */
    public ListNode leetCode24(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode next = head.next;
        ListNode resultNode = leetCode24(next.next);

        //这步是为了避免next.next = head导致的内存溢出 先将head.next断开
        head.next = null;
        next.next = head;
        head.next = resultNode;

        return next;
    }
}
