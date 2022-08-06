package com.mt.algorithm.algorithmstudy.basic;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * @Description
 * @Author T
 * @Date 2022/8/1
 */
@Service
public class PartSixBinaryTree {

    /**
     * Definition for a TreeNode.
     */
    @AllArgsConstructor
    @NoArgsConstructor
    @Data
    static class TreeNode {
        private int val;
        private TreeNode left;
        private TreeNode right;

        public TreeNode(int val) {
            this.val = val;
        }
    }

    /**
     * Definition for a Node.
     */
    @AllArgsConstructor
    @NoArgsConstructor
    @Data
    class Node {
        public int val;
        public List<Node> children;

        public Node(int _val) {
            val = _val;
        }
    }

    /**
     * 144. 二叉树的前序遍历
     *
     * @param root
     * @return
     */
    public List<Integer> leetCode144(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        preorderTraversal(root, result);
        return result;
    }

    private void preorderTraversal(TreeNode root, List<Integer> result) {
        if (root == null) {
            return;
        }
        result.add(root.val);
        preorderTraversal(root.left, result);
        preorderTraversal(root.right, result);
    }

    /**
     * 102. 二叉树的层序遍历
     *
     * @param root
     * @return
     */
    public List<List<Integer>> leetCode102(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();

        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> curList = new ArrayList<>();
            int len = queue.size();

            while (len > 0) {
                TreeNode curNode = queue.poll();
                curList.add(curNode.val);

                if (curNode.left != null) queue.add(curNode.left);
                if (curNode.right != null) queue.add(curNode.right);

                len--;
            }

            result.add(curList);
        }

        return result;
    }

    /**
     * 107. 二叉树的层序遍历 II
     * <p>
     * 给你二叉树的根节点 root ，返回其节点值 自底向上的层序遍历 。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
     *
     * @param root
     * @return
     */
    public List<List<Integer>> leetCode107(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();

        if (root == null) return result;

        Queue<TreeNode> queue = new LinkedList<>();
        Stack<List<Integer>> stack = new Stack<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> curList = new ArrayList<>();
            int len = queue.size();

            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                curList.add(curNode.val);

                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }

            stack.push(curList);
        }

        while (stack.size() > 0) {
            result.add(stack.pop());
        }

        return result;
    }

    /**
     * 199. 二叉树的右视图
     * <p>
     * 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
     *
     * @param root
     * @return
     */
    public List<Integer> leetCode199(TreeNode root) {
        List<Integer> result = new ArrayList<>();

        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> curList = new ArrayList<>();
            int len = queue.size();

            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                curList.add(curNode.val);

                if (curNode.right != null) queue.offer(curNode.right);
                if (curNode.left != null) queue.offer(curNode.left);
            }

            result.add(curList.get(0));
        }

        return result;
    }

    /**
     * 637. 二叉树的层平均值
     * <p>
     * 给定一个非空二叉树的根节点 root , 以数组的形式返回每一层节点的平均值。与实际答案相差 10-5 以内的答案可以被接受。
     *
     * @param root
     * @return
     */
    public List<Double> leetCode637(TreeNode root) {
        List<Double> result = new ArrayList<>();

        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int len = queue.size();
            double sum = 0;
            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                sum += curNode.val;
                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }

            result.add(sum / len);
        }

        return result;
    }

    /**
     * 429. N 叉树的层序遍历
     *
     * @param root
     * @return
     */
    public List<List<Integer>> leetCode429(Node root) {
        List<List<Integer>> result = new ArrayList<>();

        if (root == null) {
            return result;
        }

        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> curList = new ArrayList<>();
            int len = queue.size();

            for (int i = 0; i < len; i++) {
                Node curNode = queue.poll();
                curList.add(curNode.val);
                List<Node> children = curNode.children;
                if (children != null && children.size() > 0) {
                    for (Node node : children) {
                        queue.offer(node);
                    }
                }
            }

            result.add(curList);
        }

        return result;
    }

    /**
     * 515. 在每个树行中找最大值
     * <p>
     * 给定一棵二叉树的根节点 root ，请找出该二叉树中每一层的最大值。
     *
     * @param root
     * @return
     */
    public List<Integer> leetCode515(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> res = new ArrayList<>();
        dfs(res, root, 0);
        return res;
    }

    private void dfs(List<Integer> res, TreeNode root, int curHeight) {
        if (curHeight == res.size()) {
            res.add(root.val);
        } else {
            res.set(curHeight, Math.max(res.get(curHeight), root.val));
        }
        if (root.left != null) {
            dfs(res, root.left, curHeight + 1);
        }
        if (root.right != null) {
            dfs(res, root.right, curHeight + 1);
        }
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    class Node116 {
        public int val;
        public Node116 left;
        public Node116 right;
        public Node116 next;

        public Node116(int _val) {
            val = _val;
        }
    }

    ;

    /**
     * 116. 填充每个节点的下一个右侧节点指针
     * 与 leetCode117.填充每个节点的下一个右侧节点指针 II 解法一样
     * <p>
     * 给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
     * <p>
     * struct Node {
     * int val;
     * Node *left;
     * Node *right;
     * Node *next;
     * }
     * 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
     * <p>
     * 初始状态下，所有 next 指针都被设置为 NULL。
     *
     * @param root
     * @return
     */
    public Node116 leetCode116(Node116 root) {
        if (root == null) {
            return null;
        }

        /*
         * BFS
         */
        /*Queue<Node116> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int len = queue.size();
            Node116 last = null;
            for (int i = 0; i < len; i++) {
                Node116 curNode = queue.poll();
                curNode.next = last;
                last = curNode;
                if (curNode.right != null) queue.offer(curNode.right);
                if (curNode.left != null) queue.offer(curNode.left);
            }
        }

        return root;*/

        /*
         * DFS
         */
        List<Node116> list = new ArrayList<>();
        dfs116(root, list, 0);
        return root;
    }

    private void dfs116(Node116 root, List<Node116> list, int curHeight) {
        if (list.size() == curHeight) {
            root.next = null;
            list.add(root);
        } else {
            root.next = list.get(curHeight);
            list.set(curHeight, root);
        }

        if (root.right != null) dfs116(root.right, list, curHeight + 1);
        if (root.left != null) dfs116(root.left, list, curHeight + 1);
    }

    /**
     * 104. 二叉树的最大深度
     * <p>
     * 给定一个二叉树，找出其最大深度。
     * <p>
     * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
     * <p>
     * 说明: 叶子节点是指没有子节点的节点。
     *
     * @param root
     * @return
     */
    public int leetCode104(TreeNode root) {
        //BFS
        /*int result = 0;
        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            result++;
            int len = queue.size();

            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }
        }

        return result;*/

        //DFS bySelf
        /*int result = 0;
        if (root == null) {
            return result;
        }
        List<TreeNode> list = new ArrayList<>();
        dfs104(root, list, 0);
        return list.size();*/

        //题解DFS
        if (root == null) return 0;
        return Math.max(leetCode104(root.left), leetCode104(root.right)) + 1;
    }

    private void dfs104(TreeNode root, List<TreeNode> list, int curHeight) {
        if (list.size() == curHeight) list.add(root);
        if (root.left != null) dfs104(root.left, list, curHeight + 1);
        if (root.right != null) dfs104(root.right, list, curHeight + 1);
    }

    /**
     * 111. 二叉树的最小深度
     * <p>
     * 给定一个二叉树，找出其最小深度。
     * <p>
     * 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
     * <p>
     * 说明：叶子节点是指没有子节点的节点。
     *
     * @param root
     * @return
     */
    public int leetCode111(TreeNode root) {
        int result = 0;
        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int len = queue.size();
            result++;

            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                if (curNode.left == null && curNode.right == null) return result;
                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }
        }

        return result;
    }
}
