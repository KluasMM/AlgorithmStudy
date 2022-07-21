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

    /**
     * 19. 删除链表的倒数第 N 个结点
     * <p>
     * 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode leetCode19(ListNode head, int n) {
        /*
         * 解题思路：虚拟头结点+快慢指针
         *  慢指针要指向删除节点的上一个节点 这样就可以直接slow.next = slow.next.next;
         *  最后返回虚拟头结点的下一个结点
         */
        ListNode dummy = new ListNode(0, head);
        ListNode quick = head;
        ListNode slow = dummy;

        while (n > 0 && quick != null) {
            quick = quick.next;
            n--;
        }

        while (quick != null) {
            quick = quick.next;
            slow = slow.next;
        }

        slow.next = slow.next.next;

        return dummy.next;
    }

    /**
     * 面试题 02.07. 链表相交
     * <p>
     * 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。
     *
     * @param headA
     * @param headB
     * @return
     */
    public ListNode leetCode0207(ListNode headA, ListNode headB) {
        /*
         * 解题思路：
         *  1先找出两个链表的长度差a
         *  2然后长链表从head+a开始 短链表从head开始 同时向后走
         *  3如果存在相同的就返回结点 如果两个指针最后都为null 返回null
         *
         * 注：找长度差的时候 短链表到头了将指针指向长链表
         * 然后继续同时前进到长链接到头 将指针指向短链表 此时就达到了上述步骤2的效果
         */
        ListNode a = headA;
        ListNode b = headB;

        while (a != b) {
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }

        return a;
    }

    /**
     * 142. 环形链表 II
     * <p>
     * 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
     * <p>
     * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
     * <p>
     * 不允许修改 链表
     *
     * @param head
     * @return
     */
    public ListNode leetCode142(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        boolean noCycle = true;

        while (slow != null && fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                noCycle = false;
                break;
            }
        }

        if (noCycle) {
            return null;
        }

        fast = head;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }

        return fast;
    }
}
