package com.mt.algorithm.algorithmstudy.basic;

import org.springframework.stereotype.Service;

import java.util.*;

/**
 * @Description
 * @Author T
 * @Date 2022/8/17
 */
@Service
public class PartSevenBacktrack {

    /**
     * 77. 组合
     * <p>
     * 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
     * <p>
     * 你可以按 任何顺序 返回答案。
     *
     * @param n
     * @param k
     * @return
     */
    public List<List<Integer>> leetCode77(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        backtrack77(1, n, k, list, result);
        return result;
    }

    private void backtrack77(int i, int n, int k, List<Integer> list, List<List<Integer>> result) {
        if (list.size() == k) {
            result.add(new ArrayList<>(list));
            return;
        }

        //结束位置优化 当剩余元素不够满足k个时 结束
        for (; i <= n - (k - list.size()) + 1; i++) {
            list.add(i);
            backtrack77(i + 1, n, k, list, result);
            list.remove(list.size() - 1);
        }
    }

    /**
     * 216. 组合总和 III
     * <p>
     * 找出所有相加之和为 n 的 k 个数的组合，且满足下列条件：
     * <p>
     * 只使用数字1到9
     * 每个数字 最多使用一次 
     * 返回 所有可能的有效组合的列表 。该列表不能包含相同的组合两次，组合可以以任何顺序返回。
     *
     * @param k
     * @param n
     * @return
     */
    public List<List<Integer>> leetCode216(int k, int n) {
        /*
         * 解题思路：和leetCode77类似 都是回溯
         *  优化点：
         *    1.多一个sum参数记录当前list的总和
         *      每次撤销操作 在remove list的同时 也要对sum左减法 sum-=i
         *    2.sum > n 时 直接返回
         *
         */
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        backtrack216(1, 0, k, n, list, result);
        return result;
    }

    private void backtrack216(int i, int sum, int k, int n, List<Integer> list, List<List<Integer>> result) {
        if (sum > n) return;

        if (list.size() == k) {
            if (sum == n) {
                result.add(new ArrayList<>(list));
            }
            return;
        }

        for (; i <= 10 - (k - list.size()); i++) {
            list.add(i);
            sum += i;
            backtrack216(i + 1, sum, k, n, list, result);
            list.remove(list.size() - 1);
            sum -= i;
        }
    }

    /**
     * 17. 电话号码的字母组合
     * <p>
     * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
     * <p>
     * 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
     *
     * @param digits
     * @return
     */
    public List<String> leetCode17(String digits) {
        /*
         * 解题思路：回溯
         *  for循环对象是每次按键对应的字符串list
         *  因为是不同的按键 所以每次都是从0开始
         */
        List<String> result = new ArrayList<>();
        if (digits == null || digits.length() == 0) {
            return result;
        }
        //初始对应所有的数字，为了直接对应2-9，新增了两个无效的字符串""
        String[] numString = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        StringBuilder sb = new StringBuilder();
        //迭代处理
        backtrack17(0, digits, numString, sb, result);
        return result;
    }

    private void backtrack17(int index, String digits, String[] numString, StringBuilder sb, List<String> result) {
        if (index == digits.length()) {
            result.add(sb.toString());
            return;
        }

        //for循环的是digits的index位的数值对应的字符串list 即第几个按键的代表字符
        //这个地方要 -‘0’ 才能代表数字
        String curStr = numString[digits.charAt(index) - '0'];
        for (int i = 0; i < curStr.length(); i++) {
            sb.append(curStr.charAt(i));
            backtrack17(index + 1, digits, numString, sb, result);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    /**
     * 39. 组合总和
     * <p>
     * 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
     * <p>
     * candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
     * <p>
     * 对于给定的输入，保证和为 target 的不同组合数少于 150 个。
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> leetCode39(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Deque<Integer> list = new LinkedList<>();
        Arrays.sort(candidates);
        backtrack39(0, 0, target, candidates, list, result);
        return result;
    }

    private void backtrack39(int i, int sum, int target, int[] candidates, Deque<Integer> list, List<List<Integer>> result) {
        if (sum == target) {
            result.add(new ArrayList<>(list));
            return;
        }

        for (; i < candidates.length; i++) {
            if (sum > target) break;
            list.offerLast(candidates[i]);
            sum += candidates[i];
            backtrack39(i, sum, target, candidates, list, result);
            list.removeLast();
            sum -= candidates[i];
        }
    }

    /**
     * 40. 组合总和 II
     * <p>
     * 给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
     * <p>
     * candidates 中的每个数字在每个组合中只能使用 一次 。
     * <p>
     * 注意：解集不能包含重复的组合。 
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> leetCode40(int[] candidates, int target) {
        /*
         * 解题思路：和leetCode39的解题思路一致 重点在于去重 将解题思路理解为一个多叉树 纵向是单条结果计算 横向是下个结果计算
         *  用一个boolean[] flag来存储结点是否用到过
         *  当i > 0 && candidates[i] == candidates[i - 1]时 说明重复使用了
         *  但是重复使用有两种情况
         *      1.纵向重复使用了 也就是queue中存在同样元素 这是允许的 这种情况下重复元素已经被占用 即flag[i - 1]为true
         *      2.横向重复使用了 也就是查找下一个结果时 这时flag[]已经把占用的元素退回了 即flag[i - 1]为false
         *  另一种去重方式 就是引入index 让int i = index; 通过 i > index代替flag[i - 1] leetCode90就是这么做的
         */
        List<List<Integer>> result = new LinkedList<>();
        Deque<Integer> queue = new LinkedList<>();
        boolean[] flag = new boolean[candidates.length];

        Arrays.sort(candidates);

        backtrack40(0, 0, target, flag, candidates, queue, result);

        return result;
    }

    private void backtrack40(int i, int sum, int target, boolean[] flag, int[] candidates, Deque<Integer> queue, List<List<Integer>> result) {
        if (sum == target) {
            result.add(new ArrayList<>(queue));
            return;
        }

        for (; i < candidates.length; i++) {
            if (sum > target) break;
            if (i > 0 && candidates[i] == candidates[i - 1] && !flag[i - 1]) continue;

            queue.offerLast(candidates[i]);
            sum += candidates[i];
            //纵向占用
            flag[i] = true;

            backtrack40(i + 1, sum, target, flag, candidates, queue, result);

            queue.pollLast();
            sum -= candidates[i];
            //横向退回
            flag[i] = false;
        }
    }

    /**
     * 131. 分割回文串
     * <p>
     * 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。
     * <p>
     * 回文串 是正着读和反着读都一样的字符串。
     *
     * @param s
     * @return
     */
    public List<List<String>> leetCode131(String s) {
        List<List<String>> result = new LinkedList<>();
        Deque<String> queue = new LinkedList<>();

        backtrack131(0, s, queue, result);

        return result;
    }

    private void backtrack131(int index, String s, Deque<String> queue, List<List<String>> result) {
        if (index >= s.length()) {
            result.add(new ArrayList<>(queue));
        }

        for (int i = index; i < s.length(); i++) {
            String str = s.substring(index, i + 1);
            if (!checkStr131(str)) continue;

            queue.offerLast(str);
            backtrack131(i + 1, s, queue, result);
            queue.pollLast();
        }
    }

    /**
     * 是否是回文字符串
     *
     * @param str
     * @return
     */
    private boolean checkStr131(String str) {
        int left = 0, right = str.length() - 1;
        while (left < right) {
            if (str.charAt(left++) != str.charAt(right--)) return false;
        }
        return true;
    }

    /**
     * 93. 复原 IP 地址
     * <p>
     * 有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。
     * <p>
     * 例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。
     * 给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。你 不能 重新排序或删除 s 中的任何数字。你可以按 任何 顺序返回答案。
     *
     * @param s
     * @return
     */
    public List<String> leetCode93(String s) {
        /*
         * 解题思路和131类似
         * 重点在于[index,i + 1) 这个切割段 以及backtrack的入参index为:i + 1
         */
        List<String> result = new ArrayList<>();
        Deque<String> list = new LinkedList<>();
        backtrack93(0, s, list, result);
        return result;
    }

    private void backtrack93(int index, String s, Deque<String> list, List<String> result) {
        if (list.size() == 3) {
            String last = s.substring(index);
            if (check93(last)) {
                list.offerLast(last);
                result.add(String.join(".", new ArrayList<>(list)));
                list.pollLast();
            }
            return;
        }

        for (int i = index; i < s.length(); i++) {
            String curStr = s.substring(index, i + 1);
            if (!check93(curStr)) continue;

            list.offerLast(curStr);
            backtrack93(i + 1, s, list, result);
            list.pollLast();
        }
    }

    private boolean check93(String str) {
        int len = str.length();
        if (len > 3) return false;
        if (len > 1 && str.charAt(0) == '0') return false;
        if (Integer.parseInt(str) > 255) return false;
        return true;
    }

    /**
     * 78. 子集
     * <p>
     * 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
     * <p>
     * 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> leetCode78(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();

        Deque<Integer> queue = new LinkedList<>();
        backtrack78(0, nums, queue, result);

        return result;
    }

    private void backtrack78(int index, int[] nums, Deque<Integer> queue, List<List<Integer>> result) {
        result.add(new ArrayList<>(queue));
        //注意这里没有return 因为要取树上所有结点

        for (int i = index; i < nums.length; i++) {
            queue.offerLast(nums[i]);
            backtrack78(i + 1, nums, queue, result);
            queue.pollLast();
        }
    }

    /**
     * 90. 子集 II
     * <p>
     * 给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
     * <p>
     * 解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> leetCode90(int[] nums) {
        /*
         * 解题思路：leetCode78的升级版 去重逻辑与leetCode40相同(leetCode40是leetCode39的升级版)
         *  可以用一个boolean[]去重  nums[i] == nums[i - 1] && boolean[i-1]表示同一层已经用过 leetCode40就是这么做的
         *  也可用i > index来去重 本地就是这么做的
         */
        List<List<Integer>> result = new ArrayList<>();
        Deque<Integer> queue = new LinkedList<>();
        Arrays.sort(nums);
        backtrack90(0, nums, queue, result);
        return result;
    }

    private void backtrack90(int index, int[] nums, Deque<Integer> queue, List<List<Integer>> result) {
        result.add(new ArrayList<>(queue));
        //注意这里没有return 因为要取树上所有结点

        for (int i = index; i < nums.length; i++) {
            if (i > index && nums[i] == nums[i - 1]) continue;

            queue.offerLast(nums[i]);
            backtrack90(i + 1, nums, queue, result);
            queue.pollLast();
        }
    }

    /**
     * 491. 递增子序列
     * <p>
     * 给你一个整数数组 nums ，找出并返回所有该数组中不同的递增子序列，递增子序列中 至少有两个元素 。你可以按 任意顺序 返回答案。
     * <p>
     * 数组中可能含有重复元素，如出现两个整数相等，也可以视作递增序列的一种特殊情况。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> leetCode491(int[] nums) {
        /*
         * 解题思路:leetCode90的增强版 不能排序
         *  因为不能排序 所以用一个set去存储加入过的数字 通过判断set来进行每层的排重
         *  set写在backtrack递归方法里for循环外 这样每层都是一个新set(即横向set添加值 纵向new set)
         * 与所有子集问题一样 因为要寻找所有结果 所以不能有return
         */
        List<List<Integer>> result = new ArrayList<>();
        Deque<Integer> queue = new LinkedList<>();

        backtrack491(0, nums, queue, result);
        return result;
    }

    private void backtrack491(int index, int[] nums, Deque<Integer> queue, List<List<Integer>> result) {
        if (queue.size() > 1) result.add(new ArrayList<>(queue));

        Set<Integer> set = new HashSet<>();
        for (int i = index; i < nums.length; i++) {
            if (queue.size() > 0) {
                Integer last = queue.peekLast();
                if (nums[i] < last) continue;
            }
            if (set.contains(nums[i])) continue;
            set.add(nums[i]);
            queue.offerLast(nums[i]);
            backtrack491(i + 1, nums, queue, result);
            queue.pollLast();
        }
    }

    /**
     * 46. 全排列
     * <p>
     * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> leetCode46(int[] nums) {
        /*
         * 解题思路：
         *  不在需要index记录位置了 每层都要从遍历 但是不能包含已存在的元素
         */
        List<List<Integer>> result = new ArrayList<>();
        Deque<Integer> queue = new LinkedList<>();
        backtrack46(nums, queue, result);
        return result;
    }

    private void backtrack46(int[] nums, Deque<Integer> queue, List<List<Integer>> result) {
        if (queue.size() == nums.length) result.add(new ArrayList<>(queue));

        for (int i = 0; i < nums.length; i++) {
            if (queue.contains(nums[i])) continue;
            queue.offerLast(nums[i]);
            backtrack46(nums, queue, result);
            queue.pollLast();
        }
    }

    /**
     * 47. 全排列 II
     * <p>
     * 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> leetCode47(int[] nums) {
        /*
         * 解题思路：leetCode46的增强版 存在重复元素
         *  用hashSet做每层的去重 原理与leetCode491一样
         * 扩展：看一下卡尔的去重逻辑
         *      // used[i - 1] == true，说明同⼀树⽀nums[i - 1]使⽤过
                // used[i - 1] == false，说明同⼀树层nums[i - 1]使⽤过
                // 如果同⼀树层nums[i - 1]使⽤过则直接跳过
                if (i > 0 && nums[i] == nums[i - 1] && used[i - 1] == false) {
                    continue;
                }
         */
        List<List<Integer>> result = new ArrayList<>();
        Deque<Integer> queue = new LinkedList<>();
        boolean[] used = new boolean[nums.length];
        backtrack47(nums, used, queue, result);
        return result;
    }

    private void backtrack47(int[] nums, boolean[] used, Deque<Integer> queue, List<List<Integer>> result) {
        if (queue.size() == nums.length) result.add(new ArrayList<>(queue));

        Set<Integer> set = new HashSet<>(nums.length);
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) continue;
            if (set.contains(nums[i])) continue;

            set.add(nums[i]);
            queue.offerLast(nums[i]);
            used[i] = true;
            backtrack47(nums, used, queue, result);
            queue.pollLast();
            used[i] = false;
        }
    }
}
