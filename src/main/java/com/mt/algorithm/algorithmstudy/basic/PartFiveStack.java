package com.mt.algorithm.algorithmstudy.basic;

import org.springframework.stereotype.Service;

import java.util.*;

/**
 * @Description
 * @Author T
 * @Date 2022/7/31
 */
@Service
public class PartFiveStack {

    /**
     * 20. 有效的括号
     * <p>
     * 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
     * <p>
     * 有效字符串需满足：
     * <p>
     * 左括号必须用相同类型的右括号闭合。
     * 左括号必须以正确的顺序闭合。
     *
     * @param s
     * @return
     */
    public boolean leetCode20(String s) {
        Stack<Character> stack = new Stack<>();

        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ')') {
                if (stack.isEmpty() || stack.pop() != '(') {
                    return false;
                }
            } else if (s.charAt(i) == ']') {
                if (stack.isEmpty() || stack.pop() != '[') {
                    return false;
                }
            } else if (s.charAt(i) == '}') {
                if (stack.isEmpty() || stack.pop() != '{') {
                    return false;
                }
            } else {
                stack.push(s.charAt(i));
            }
        }

        return stack.isEmpty();
    }

    /**
     * 1047. 删除字符串中的所有相邻重复项
     * <p>
     * 给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。
     * <p>
     * 在 S 上反复执行重复项删除操作，直到无法继续删除。
     * <p>
     * 在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。
     *
     * @param s
     * @return
     */
    public String leetCode1047(String s) {
        int len = s.length();
        List<Character> list = new ArrayList<>(len);
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (!stack.isEmpty() && stack.peek() == s.charAt(i)) {
                stack.pop();
            } else {
                stack.push(s.charAt(i));
            }
        }

        StringBuilder str = new StringBuilder();
        while (!stack.isEmpty()) {
            str.insert(0, stack.pop());
        }

        return str.toString();
    }

    /**
     * 150. 逆波兰表达式求值
     * <p>
     * 有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
     * <p>
     * 注意 两个整数之间的除法只保留整数部分。
     * <p>
     * 可以保证给定的逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
     *
     * @param tokens
     * @return
     */
    public int leetCode150(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < tokens.length; i++) {
            String cur = tokens[i];
            if (cur.equals("+")) {
                Integer one = stack.pop();
                Integer two = stack.pop();
                stack.push(one + two);
            } else if (cur.equals("-")) {
                Integer one = stack.pop();
                Integer two = stack.pop();
                stack.push(two - one);
            } else if (cur.equals("*")) {
                Integer one = stack.pop();
                Integer two = stack.pop();
                stack.push(one * two);
            } else if (cur.equals("/")) {
                Integer one = stack.pop();
                Integer two = stack.pop();
                stack.push(two / one);
            } else {
                stack.push(Integer.parseInt(cur));
            }
        }

        return stack.pop();
    }

    /**
     * 239. 滑动窗口最大值
     * <p>
     * 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
     * <p>
     * 返回 滑动窗口中的最大值 。
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] leetCode239(int[] nums, int k) {
        /*
         * 解题思路1：单调队列
         *  使用双端队列deque 存储滑动窗口索引
         *  保证deque中 索引由小到大 并且索引对应的元素由大到小
         *  数据插入：窗口移动时，向队尾插入新增元素（以下比较的都是索引对应元素，实际插入的是索引）
         *      如果新增元素比队尾元素大 则队尾元素出队 直到比队尾元素小 插入该元素的索引到队尾
         *  数据减少：窗口移动时，离开元素的索引与deque队首元素比较
         *      如果离开元素的索引大于等于deque队首值 说明当前deque的最大值已经不在窗口内了
         *      所以需要deque队首出队 直到离开元素的索引小于deque队首元素
         *      这时 将队首元素（索引）的值加入结果集
         */
        Deque<Integer> deque = new LinkedList<>();
        //前k个元素 执行数据插入
        for (int i = 0; i < k; i++) {
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.pollLast();
            }
            deque.offerLast(i);
        }

        int len = nums.length;
        int[] result = new int[len - k + 1];
        //第一个元素就是队首
        result[0] = nums[deque.peekFirst()];

        for (int i = k; i < len; i++) {
            //第i个元素 执行数据插入
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.pollLast();
            }
            deque.offerLast(i);

            //窗口左侧移除元素索引 执行数据减少
            int remove = i - k;
            while (!deque.isEmpty() && remove >= deque.peekFirst()) {
                deque.pollFirst();
            }

            //此时队首的值 就是结果
            result[i - k + 1] = nums[deque.peekFirst()];
        }

        return result;
    }

    /**
     * 347. 前 K 个高频元素
     * <p>
     * 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] leetCode347(int[] nums, int k) {
        int[] result = new int[k];

        Map<Integer, Integer> map = new HashMap<>(nums.length);
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }

        Set<Map.Entry<Integer, Integer>> entrySet = map.entrySet();

        PriorityQueue<Map.Entry<Integer, Integer>> priorityQueue = new PriorityQueue<>(
                (o1, o2) -> o2.getValue() - o1.getValue()
        );

        for (Map.Entry<Integer, Integer> entry : entrySet) {
            priorityQueue.offer(entry);
        }

        for (int i = 0; i < k; i++) {
            result[i] = priorityQueue.poll().getKey();
        }

        return result;
    }
}
